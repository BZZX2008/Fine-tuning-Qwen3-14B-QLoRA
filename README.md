# fine-tuning-Qwen3-14B-QLoRA
**项目内容**:  医疗垂直领域，提升大模型在医疗方面的专业回复能力。


**模型选择**：Qwen3-14B


**微调方式**：QLoRA

**微调对象**：注意力层参数：q_pro, k_pro，v_pro, o_pro
             前馈神经网络层参数：gate_pro, up_pro, down_pro

**数据集**：中文医疗对话数据集https://github.com/Toyhom/Chinese-medical-dialogue-data。

            **包含六个科室的医疗对话**： 
                                    内科： 220606个问答对
                                    外科： 115991个问答对
                                    妇产科： 183751个问答对
                                    肿瘤科： 75553个问答对
                                    男科： 94596个问答对
                                    儿科： 101602个问答对
                        
             **本次微调数据集**：对整个中文医疗对话数据集采用均衡采样，每个科室采样3000个问答对。一共六个科室，微调数据集总样本数为18000条
             
             **取验证集**： 按照15%获取。 18000条 * 0.15=2700条
             
             **训练集**： 18000条-2700条=15300条


**GPU环境**：RTX4090 24G

⏱️  训练耗时: 23785 秒

📊 **下面是每次epoch的验证损失**: 

                      epoch=1时，验证损失
                      {'eval_loss': '2.008', 'eval_runtime': '506.6', 'eval_samples_per_second': '5.329', 'eval_steps_per_second': '5.329', 'epoch': '1'}                                                                                                                                                                            
                      33%|  | 1913/5739 [2:11:02<3:30:55,  3.31s/it

                      epoch=2时，验证损失
                      {'eval_loss': '1.966', 'eval_runtime': '506.3', 'eval_samples_per_second': '5.333', 'eval_steps_per_second': '5.333', 'epoch': '2'}                                                                                                                                                                            
                      67%|  | 3826/5739 [4:22:59<1:46:32,  3.34s/it

                      epoch=3时，验证损失
                      {'eval_loss': '2.195', 'eval_runtime': '510.8', 'eval_samples_per_second': '5.285', 'eval_steps_per_second': '5.285', 'epoch': '3'}                                                                                                                                                                            
                      100%|  | 5739/5739 [6:36:22<00:00,  3.26s/it

其中epoch=2时，效果最好


                                                                                                                                                                        
