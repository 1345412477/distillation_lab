# 多模态图片识别蒸馏 - 研究资料整理

> 收集时间：2026-08-03。围绕"图片识别能力蒸馏 + 实验调参 + 评估对比"整理。

## 1. Qwen3-VL / Qwen3.5-VL 微调与蒸馏实践

- **主流做法是 LoRA / QLoRA**：冻结视觉编码器与投影层，只对语言模型部分做 LoRA（本项目已按此实现），可在单卡（RTX 4090D 级别）高效训练。
  - 来源：CSDN《基于Qwen3-VL-WEBUI的多模态微调全流程解析》、星图 GPU 平台 Qwen3-VL-8B LoRA 微调教程
- **LoRA 参数惯例**：
  - rank 16~64（视觉任务常用 16/32），alpha 通常为 rank 的 2 倍；
  - 视觉-语言任务学习率一般比纯文本低：1e-5 ~ 2e-5 常见，本项目默认 5e-5 可下调；
  - epoch 1~3 即可，数据量小时过多轮次容易过拟合。
- **多模态蒸馏的特殊性**（来源：arXiv 2505.10526 及 ICCV 2021 视觉语言蒸馏综述）：
  - 图像特征维度高、视觉与文本表征对齐难、同一图片的人类描述方式多样；
  - 教师用"回答"做离线数据蒸馏（本项目路径）是轻量可行方案；进阶可做 logit 级蒸馏或
    冻结视觉编码器 + LoRA 语言模型后，再联合微调视觉适配器（级联微调，来源：小红书多模态微调笔记）。

## 2. 图片描述/识别任务的评估指标

| 指标 | 类型 | 说明 |
|------|------|------|
| BLEU | 参考依赖 | n-gram 精确率 + 长度惩罚，衡量与参考答案的词汇重合 |
| ROUGE-L | 参考依赖 | 基于最长公共子序列的 F1，对长句更鲁棒 |
| METEOR / CIDEr / SPICE | 参考依赖 | 语义/共识度更强的图片描述指标 |
| CLIPScore | 免参考 | 图像-文本嵌入相似度，与人工判断相关性最高 |
| CHAIR | 幻觉检测 | 检测描述中不存在的物体 |

- 参考：ACL/ECNLP 2024、IEEE 2026 图片描述综述（BLEU/METEOR/ROUGE-L/CIDEr/SPICE/CLIPScore/BERTScore）。
- 本项目离线环境优先实现**无外部依赖**的 BLEU-1/ROUGE-L + 字符/词级 F1，作为"内容覆盖率"维度；
  若后续引入 CLIP 系列模型，可加免参考 CLIPScore。

## 3. 实验调参覆盖清单（应全部暴露到 UI）

### 数据生成（教师）
- 生成数量、prompts_per_image（每图多提示，数据增强）、温度、top_p、top_k、
  repetition_penalty、num_beams、max_new_tokens、enable_think（Qwen 思考模式）

### 训练（学生）
- LoRA：rank、alpha、dropout、target_modules（默认全线性层）
- 优化器：AdamW / AdamW8bit / SGD；learning_rate、weight_decay、warmup_steps、
  grad_clip、grad_accum_steps、batch_size、epochs、max_seq_len、seed
- 硬件：4-bit 量化开关

### 评估
- 评估问题/参考答案、eval_max_new_tokens、多模态评估图片数

### 剪枝（已有）
- 策略（FFN/注意力头/层）、比例、重要性指标（L1/L2）
- 恢复：开关、epochs、lr，以及恢复用 LoRA rank/alpha/dropout

## 4. 三模型对比设计

蒸馏实验完成后应保存并对比：
1. **教师模型**（teacher_results：同一评估集上的回答）
2. **蒸馏前学生**（baseline）
3. **蒸馏后学生**（after）

对比维度沿用评估引擎的 4 维评分（完整性/覆盖率/流畅度/一致性），
多模态任务按"问题 × 图片"逐条对比，表格展示三列回答。
