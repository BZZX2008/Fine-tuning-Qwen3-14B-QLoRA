#!/usr/bin/env python3
"""
Qwen3-14B 医疗对话 LoRA 微调（训练/验证/测试集划分）
- 每科室采样3000条，共18000条
- 80%训练 / 10%验证 / 10%测试（测试集不参与训练）
环境：torch 2.10.0+cu130, transformers 5.4.0, bitsandbytes 0.49.2
单卡 RTX 4090 24GB
"""

import os, sys, types, importlib.machinery, random, json, warnings
from enum import Enum
from collections import defaultdict
warnings.filterwarnings("ignore")

# ===================== 0. 运行时环境修复 =====================
if 'OMP_NUM_THREADS' in os.environ:
    try: int(os.environ['OMP_NUM_THREADS'])
    except ValueError: del os.environ['OMP_NUM_THREADS']

CUDA_LIB = '/root/miniconda3/envs/awq_quant/lib/python3.10/site-packages/nvidia/cu13/lib'
if os.path.isdir(CUDA_LIB) and CUDA_LIB not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = CUDA_LIB + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    print(f"[环境] 已注入 CUDA 13 库路径: {CUDA_LIB}")

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
print("[环境] 已安装通用 torchvision 假模块拦截器")

# ===================== 1. 正式导入 =====================
import torch, pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

print(f"PyTorch 版本: {torch.__version__}")

# ===================== 2. 配置参数 =====================
MODEL_PATH = "/root/autodl-tmp/Qwen3-14B"
DATA_DIR   = "data"
OUTPUT_DIR = "/root/autodl-tmp/qwen3-14b-medical-final"
EVAL_JSONL = "eval_dataset.jsonl"       # 验证集
TEST_JSONL = "test_dataset.jsonl"       # 测试集（最终评估用）

MAX_SEQ_LENGTH        = 2048
LORA_RANK             = 128
LORA_ALPHA            = 256
LORA_DROPOUT          = 0.1
BATCH_SIZE            = 1
GRADIENT_ACCUMULATION = 8
LEARNING_RATE         = 2e-4
NUM_EPOCHS            = 3                     # 可根据需要调整

SAMPLES_PER_DEPT = 3000                      # 每个科室采样 3000 条
MAX_Q_LEN        = 600
MAX_A_LEN        = 600
TEST_RATIO       = 0.10                      # 测试集占比
VALID_RATIO      = 0.1111                    # 验证集占剩余数据的比例（使得总验证集≈10%）
RANDOM_SEED      = 42

# ===================== 3. 加载基础模型 + LoRA =====================
print("正在加载本地 Qwen3-14B 模型（4‑bit）...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH, max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True, trust_remote_code=True, local_files_only=True,
)

model = FastLanguageModel.get_peft_model(
    model, r=LORA_RANK,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
    use_gradient_checkpointing="unsloth", random_state=RANDOM_SEED,
)

# ===================== 4. 数据加载 & 均衡采样 =====================
def read_csv_with_encoding(file_path):
    for enc in ['gbk','gb2312','gb18030','utf-8']:
        try: return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError: continue
    raise ValueError(f"无法读取: {file_path}")

def load_medical_data(data_dir, samples_per_dept=None, max_q=500, max_a=500, seed=42):
    data = []
    departments = {
        'IM_内科':'内科','Surgical_外科':'外科','Pediatric_儿科':'儿科',
        'Oncology_肿瘤科':'肿瘤科','OAGD_妇产科':'妇产科','Andriatria_男科':'男科'
    }
    for dept_dir, dept_name in departments.items():
        dept_path = os.path.join(data_dir, dept_dir)
        if not os.path.exists(dept_path): continue
        for fname in os.listdir(dept_path):
            if not fname.endswith('.csv'): continue
            file_path = os.path.join(dept_path, fname)
            try:
                df = read_csv_with_encoding(file_path)
                for _, row in df.iterrows():
                    q = str(row.get('question', row.get('ask', '')).strip())
                    a = str(row.get('answer', row.get('response', '')).strip())
                    if q and a and len(q) <= max_q and len(a) <= max_a:
                        data.append({'question':q,'answer':a,'department':dept_name})
            except: pass
    print(f"✅ 原始有效数据: {len(data)} 条")

    if samples_per_dept:
        random.seed(seed)
        dept_data = defaultdict(list)
        for item in data: dept_data[item['department']].append(item)
        sampled = []
        for dept, items in dept_data.items():
            n = min(samples_per_dept, len(items))
            sampled.extend(random.sample(items, n))
            print(f"  {dept}: 采样 {n} 条")
        random.shuffle(sampled)
        print(f"均衡采样完成: {len(sampled)} 条")
        data = sampled
    return Dataset.from_list(data)

def formatting_prompts_func(examples):
    texts = []
    for q, a in zip(examples['question'], examples['answer']):
        messages = [{"role":"user","content":q},{"role":"assistant","content":a}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        texts.append(text)
    return {"text": texts}

# 加载并采样原始数据（每个科室3000条）
dataset = load_medical_data(DATA_DIR, samples_per_dept=SAMPLES_PER_DEPT,
                            max_q=MAX_Q_LEN, max_a=MAX_A_LEN, seed=RANDOM_SEED)

# 首先分出测试集（10%）
dataset_split = dataset.train_test_split(test_size=TEST_RATIO, seed=RANDOM_SEED)
test_dataset = dataset_split['test']
remaining = dataset_split['train']

# 再从剩余数据中分出验证集（占比 ~10%）
remaining_split = remaining.train_test_split(test_size=VALID_RATIO, seed=RANDOM_SEED)
train_dataset = remaining_split['train']
eval_dataset  = remaining_split['test']

print(f"训练集: {len(train_dataset)} 条")
print(f"验证集: {len(eval_dataset)} 条")
print(f"测试集: {len(test_dataset)} 条")

# 格式化数据
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
eval_dataset  = eval_dataset.map(formatting_prompts_func, batched=True)
test_dataset  = test_dataset.map(formatting_prompts_func, batched=True)

# 保存验证集和测试集为 JSONL（注意测试集也保存原始文本）
def save_jsonl(dataset, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(json.dumps({"question":ex["question"],"answer":ex["answer"]}, ensure_ascii=False)+"\n")

save_jsonl(eval_dataset, EVAL_JSONL)
save_jsonl(test_dataset, TEST_JSONL)
print("✅ 验证集和测试集已保存")

# ===================== 5. 训练参数 =====================
training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    warmup_steps=100,                           # 稍微加大预热量
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=False, bf16=True,
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=RANDOM_SEED,
    output_dir=OUTPUT_DIR,
    report_to="tensorboard",
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,                # 训练结束后自动加载最优模型
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=train_dataset, eval_dataset=eval_dataset,
    dataset_text_field="text", max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2, packing=False, args=training_args,
)

# ===================== 6. 开始训练 =====================
gpu_stats = torch.cuda.get_device_properties(0)
print(f"🎮 GPU: {gpu_stats.name}, 总显存: {gpu_stats.total_memory / 1024**3:.1f} GB")
print("🚀 开始训练...")
trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved()/1024**3, 3)
print(f"⏱️  训练耗时: {trainer_stats.metrics['train_runtime']:.0f} 秒")
print(f"📊 峰值显存占用: {used_memory} GB")
print(f"🔍 最佳验证损失: {trainer_stats.metrics.get('eval_loss', 'N/A')}")

# ===================== 7. 保存最终模型 =====================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"💾 模型已保存至 {OUTPUT_DIR}")

# ===================== 8. 推理测试（简单演示） =====================
FastLanguageModel.for_inference(model)
def generate_response(question):
    messages = [{"role":"user","content":question}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token_id = 0
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9,
                             repetition_penalty=1.1, do_sample=True,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.replace("<think>","").replace("</think>","").strip()

test_questions = [
    "我最近总是感觉头晕，应该怎么办？",
    "感冒发烧应该吃什么药？",
    "高血压患者需要注意什么？"
]
for q in test_questions:
    print(f"\n{'='*50}\n👤 问题：{q}\n🤖 回答：{generate_response(q)}")
print("\n✨ 训练完成！")
