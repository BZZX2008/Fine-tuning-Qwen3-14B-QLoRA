# fine-tuning-Qwen3-14B-QLoRA

中文医疗对话数据集：https://github.com/Toyhom/Chinese-medical-dialogue-data。

   • <Andriatria_男科> 94596个问答对


   • <IM_内科> 220606个问答对


   • <OAGD_妇产科> 183751个问答对


   • <Oncology_肿瘤科> 75553个问答对


   • <Pediatric_儿科> 101602个问答对


   • <Surgical_外科> 115991个问答对

数据均衡采样：每个科室3000个问答队，一共六个科室。总数据样本18000条


验证集：按照15%获取。18000条 * 0.15=2700条


GPU环境：RTX4090 24G

⏱️  训练耗时: 23785 秒
📊 下面是每次epoch的验证损失: 

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


                                                                                                                                                                        
