#!/usr/bin/env python3
"""
医疗模型评估 —— 最终稳定版（BERTScore CPU 自适应截断）
Qwen 推理：GPU
BERTScore：CPU，batch_size=16，内部 tokenizer 自动截断 256 token
依赖：pip install jieba rouge sacrebleu bert-score tqdm modelscope
"""

import os, sys, types, importlib.machinery, json, warnings
from enum import Enum
warnings.filterwarnings("ignore")

# ── 0. 环境修复（保持与训练脚本一致） ──
if 'OMP_NUM_THREADS' in os.environ:
    try: int(os.environ['OMP_NUM_THREADS'])
    except ValueError: del os.environ['OMP_NUM_THREADS']

CUDA_LIB = '/root/miniconda3/envs/awq_quant/lib/python3.10/site-packages/nvidia/cu13/lib'
if os.path.isdir(CUDA_LIB) and CUDA_LIB not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = CUDA_LIB + ':' + os.environ.get('LD_LIBRARY_PATH', '')

class FakeInterpolationMode(Enum):
    NEAREST = 0; NEAREST_EXACT = 1; BILINEAR = 2; BICUBIC = 3
    BOX = 4; HAMMING = 5; LANCZOS = 6

class FakeTorchvisionLoader:
    def find_spec(self, fullname, path, target=None):
        if fullname == "torchvision" or fullname.startswith("torchvision."):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__version__ = "0.0.0"; mod.__spec__ = spec; mod.__path__ = []
        if spec.name == "torchvision.transforms":
            mod.InterpolationMode = FakeInterpolationMode
        return mod
    def exec_module(self, module): pass

sys.meta_path.insert(0, FakeTorchvisionLoader())

# ── 1. 导入 ──
import torch
from unsloth import FastLanguageModel
import jieba
from rouge import Rouge
from tqdm import tqdm
from bert_score import BERTScorer
from modelscope import snapshot_download
from transformers import AutoTokenizer

# ── 2. 配置 ──
LORA_PATH = "/root/autodl-tmp/qwen3-14b-medical-final/"
EVAL_JSONL = "eval_dataset_final.jsonl"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 8                     # Qwen推理批量

BERT_MODEL_NAME = "tiansz/bert-base-chinese"
BERT_CACHE_DIR = "/root/autodl-tmp/models/bert-score"
BERT_MAX_TOKENS = 256              # BERT 截断长度

# ── 3. 下载 BERT 模型（仅首次） ──
print("正在从 ModelScope 下载 bert-base-chinese（仅首次下载）...")
bert_model_path = snapshot_download(BERT_MODEL_NAME, cache_dir=BERT_CACHE_DIR)
print('bert-base-chinese的位置：'+ str(bert_model_path))
# ── 4. 加载微调 Qwen 模型 ──
print("加载微调后的 Qwen 模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA_PATH, max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True, trust_remote_code=True, local_files_only=True,
)
FastLanguageModel.for_inference(model)
if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.pad_token_id = 0

# ── 5. 加载验证集 ──
with open(EVAL_JSONL, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]
questions = [r["question"] for r in records]
references = [r["answer"] for r in records]
print(f"验证样本数: {len(questions)}")

# ── 6. 批量生成预测（GPU） ──
def generate_batch(qs):
    msgs = []
    for q in qs:
        msgs.append(tokenizer.apply_chat_template(
            [{"role":"user","content":q}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False))
    inputs = tokenizer(msgs, return_tensors="pt", padding=True,
                       truncation=True, max_length=MAX_SEQ_LENGTH).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=256, temperature=0.7, top_p=0.9,
            repetition_penalty=1.1, do_sample=True,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    res = []
    for i, out in enumerate(outputs):
        text = tokenizer.decode(out[inputs.input_ids.shape[1]:], skip_special_tokens=True)
        res.append(text.replace("<think>","").replace("</think>","").strip())
    return res

predictions = []
for i in tqdm(range(0, len(questions), BATCH_SIZE), desc="批量推理中"):
    predictions.extend(generate_batch(questions[i:i+BATCH_SIZE]))

# ── 7. 数据清洗 ──
def clean(text):
    t = text.strip()
    if not t or all(ch in '！？。；“”’‘…—～　 \t\n\r' for ch in t): return ""
    return t

clean_refs, clean_preds = [], []
for ref, pred in zip(references, predictions):
    r, p = clean(ref), clean(pred)
    if r and p:
        clean_refs.append(r); clean_preds.append(p)
print(f"有效样本: {len(clean_refs)}")

# ── 8. sacreBLEU ──
try:
    from sacrebleu import corpus_bleu
    bleu = corpus_bleu(clean_preds, [clean_refs]).score
except:
    bleu = None

# ── 9. ROUGE ──
def tokenize(text):
    return " ".join([w.strip() for w in jieba.cut(text) if w.strip()])

try:
    rouge = Rouge()
    r_tok = [tokenize(t) for t in clean_refs]
    p_tok = [tokenize(t) for t in clean_preds]
    scores = rouge.get_scores(p_tok, r_tok, avg=True)
    r1, r2, rl = scores['rouge-1']['f']*100, scores['rouge-2']['f']*100, scores['rouge-l']['f']*100
except:
    r1=r2=rl=0

# ── 10. BERTScore（CPU，自适应截断） ──
avg_bertscore = None
try:
    # 预先准备自定义 tokenizer（已截断）
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
    bert_tokenizer.model_max_length = BERT_MAX_TOKENS

    # 检测 BERTScorer 是否支持 tokenizer 参数
    import inspect
    sig = inspect.signature(BERTScorer.__init__)
    if 'tokenizer' in sig.parameters:
        print("BERTScorer 支持 tokenizer 参数，直接传入...")
        scorer = BERTScorer(
            model_type=bert_model_path,
            lang="zh",
            num_layers=12,
            tokenizer=bert_tokenizer,   # 传入自定义 tokenizer
            device="cpu",
            batch_size=16,
        )
    else:
        print("BERTScorer 不支持 tokenizer 参数，修改内部 _tokenizer...")
        scorer = BERTScorer(
            model_type=bert_model_path,
            lang="zh",
            num_layers=12,
            device="cpu",
            batch_size=16,
        )
        scorer._tokenizer = bert_tokenizer   # 直接替换内部 tokenizer

    # 验证截断是否生效
    test_ids = scorer._tokenizer.encode("测试 " * 500, truncation=True)
    assert len(test_ids) <= BERT_MAX_TOKENS, f"截断失败，长度 {len(test_ids)}"
    print(f"✅ BERT tokenizer 截断验证通过（测试长度 {len(test_ids)}）")

    print("计算 BERTScore（CPU 模式，可能需要几分钟）...")
    P, R, F1 = scorer.score(clean_preds, clean_refs)
    avg_bertscore = F1.mean().item() * 100
except Exception as e:
    print(f"⚠️ BERTScore 计算失败: {e}")

# ── 11. 输出结果 ──
print(f"\n📊 评估结果（有效样本 {len(clean_refs)} 条）")
if bleu: print(f"sacreBLEU (corpus): {bleu:.2f}%")
print(f"ROUGE-1 F1: {r1:.2f}%")
print(f"ROUGE-2 F1: {r2:.2f}%")
print(f"ROUGE-L F1: {rl:.2f}%")
if avg_bertscore is not None:
    print(f"BERTScore-F1 (中文): {avg_bertscore:.2f}%")
else:
    print("BERTScore-F1: 计算失败")

print(f"\n📝 前 3 条生成样例")
for i in range(min(3, len(clean_refs))):
    print(f"\nQ{i+1}: {questions[i][:80]}...")
    print(f"Ref : {clean_refs[i][:100]}...")
    print(f"Gen : {clean_preds[i][:100]}...")

print("\n✨ 评估完成！")