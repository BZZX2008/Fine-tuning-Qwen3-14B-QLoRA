# Fine-tuning-Qwen3-14B-QLoRA
**项目内容**:  医疗垂直领域，提升大模型在医疗咨询的专业回复能力。


**模型选择**：Qwen3-14B


**微调方式**：QLoRA

**微调对象**：注意力层参数：q_proj, k_proj，v_proj, o_proj
             前馈神经网络层参数：gate_proj, up_proj, down_proj

**配置参数**：
 
             MAX_SEQ_LENGTH        = 2048      #序列长度
             LORA_RANK             = 128       #LoRA的秩
             LORA_ALPHA            = 256       #缩放因子 （通常取LORA_RANK的2倍）       缩放比例=LORA_ALPHA/LORA_RANK，过小：微调效果不明显；过大：偏离原始模型，导致遗忘
             LORA_DROPOUT          = 0.1       #丢包率，为了防止过拟合
             BATCH_SIZE            = 1         #可以根据显存调整
             GRADIENT_ACCUMULATION = 8         #梯度下降
             LEARNING_RATE         = 2e-4      #学习率
             NUM_EPOCHS            = 3         #可根据需要调整，防止过拟合

             SAMPLES_PER_DEPT = 3000                      # 每个科室采样 3000 条
             MAX_Q_LEN        = 600
             MAX_A_LEN        = 600
             TEST_RATIO       = 0.10                      # 测试集占比
             VALID_RATIO      = 0.1111                    # 验证集占剩余数据的比例（使得总验证集≈10%）
             RANDOM_SEED      = 42

**数据集**：中文医疗对话数据集https://github.com/Toyhom/Chinese-medical-dialogue-data。

            **包含六个科室的医疗对话**： 
                                    内科： 220606个问答对
                                    外科： 115991个问答对
                                    妇产科： 183751个问答对
                                    肿瘤科： 75553个问答对
                                    男科： 94596个问答对
                                    儿科： 101602个问答对
                        
             **本次微调数据集**：对整个中文医疗对话数据集采用均衡采样，每个科室采样3000个问答对。一共六个科室，微调数据集总样本数为18000条
             
             **取验证集**： 按照10%获取。 18000条 * 0.10=1800条
             
             **取测试集**： 按照10%获取。 18000条 * 0.10=1800条

             **取训练集**： 按照80%获取。 18000条 - 3600条 = 14400条


**GPU环境**：RTX4090 24G          **峰值显存占用**: 22.467 GB

⏱️  **训练耗时**: 22279 秒

📊 **每次epoch的验证损失**: 其中epoch=2时，效果最好

                      epoch=1时，验证损失
                      {'eval_loss': '2.062', 'eval_runtime': '341.5', 'eval_samples_per_second': '5.271', 'eval_steps_per_second': '5.271', 'epoch': '1'} 33%| | 1800/5400 [2:03:11<3:54:37,  3.91s/it

                      epoch=2时，验证损失
                      {'eval_loss': '2.012', 'eval_runtime': '340.3', 'eval_samples_per_second': '5.289', 'eval_steps_per_second': '5.289', 'epoch': '2'} 67%|  | 3600/5400 [4:06:56<2:02:52,  4.10s/it

                      epoch=3时，验证损失
                      {'eval_loss': '2.345', 'eval_runtime': '341.1', 'eval_samples_per_second': '5.277', 'eval_steps_per_second': '5.277', 'epoch': '3'} 100%|  | 5400/5400 [6:11:14<00:00,  4.13s/it




**测试集评估结果**

           在下图中可以看到：
                     指标	                   数值	        说明
                     
                     BERTScore-F1(中文)	    70.06%	      语义相似度良好
                     
                     ROUGE-1 F1	            22.61%	      关键词覆盖良好
                     
                     ROUGE-2 F1	            6.18%	        短语匹配有限，但医疗同义替换常见
                     
                     ROUGE-L F1	            16.27%	      长句结构相似度较好
                     
                     sacreBLEU	            0.37%	        极低，符合医疗领域特点
                                                                                                                                                                                       
<img width="3059" height="1024" alt="Image" src="https://github.com/user-attachments/assets/a2f392d8-f51b-4bc0-a54f-6aacf778b889" />


**进一步优化方向**

    升级基座模型为带Instruct指令微调模型
    
    引入 DPO 对齐提升安全性与格式

    增加数据多样性（如问诊对话、检查报告解读）

    加入用户反馈，人工评估
