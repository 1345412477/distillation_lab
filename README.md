# DistillationLab 模型蒸馏与剪枝实验平台

一个基于 **PyQt6** 的桌面端模型压缩实验平台，专注于**多模态图片识别能力的知识蒸馏**与结构化剪枝，支持实验全流程的可视化配置、实时监控、多模型对比与自动评估。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ 功能特性

| 模块 | 说明 |
|------|------|
| **模型管理** | 从 ModelScope / HuggingFace / 本地路径导入模型，自动解析参数量、架构、文件大小；支持从实验输出导入 |
| **多模态蒸馏** | 教师模型生成图文数据（每图多提示词数据增强）→ 学生模型 LoRA 微调 → 蒸馏前后自动评估，全链路真实跑通 |
| **结构化剪枝** | FFN 维度 / 注意力头 / 整层三种策略，L1/L2 重要性指标，**真实重建层并更新 config**（参数与文件同步缩小），支持 LoRA 恢复微调 |
| **全参数调优** | 生成、训练、评估、剪枝 30+ 参数全部可调；实验模板一键保存/复用；批量交叉矩阵（模板 × 学生模型）顺序运行 |
| **实时监控** | 阶段进度条、逐样本识别预览（图片缩略图）、训练速度/耗时/ETA、Loss 曲线、GPU 状态 |
| **多模型对比** | 教师 / 蒸馏前 / 蒸馏后 三模型逐题对比：雷达图、维度评分表、逐图识别详情、Markdown 报告 |
| **自动评估** | 回答完整性 / 内容覆盖率（F1+BLEU-1+ROUGE-L）/ 语言流畅度 / 基线一致性 四维评分，可选 CLIP 图像对齐评分 |
| **显卡自动识别** | nvidia-smi 全字段（驱动、CUDA、算力、显存、利用率、温度、功耗）+ 详情对话框 + 首页硬件摘要 |
| **可复现** | 随机种子、完整实验配置落库、每次实验自动生成评估报告 |

---

## 🚀 快速开始

### 方式一：直接使用发布包（无需 Python / 无需联网）

| 版本 | 位置 | 说明 |
|------|------|------|
| **安装包** | `dist/DistillationLab-安装包/` | 双击 `安装程序.exe` 安装，自动创建桌面快捷方式，含卸载功能 |
| **免安装版** | `dist/DistillationLab/` | 整个文件夹复制即用，双击 `DistillationLab.exe` |
| **便携版** | `dist/DistillationLab便携版/` | 启动器会自动安装依赖（无 Python 时自动下载配置 Python 3.11） |

> 发布包由 `build_exe.py`（全量）、`build_installer.py`（安装包）、`launcher.py`（便携）构建，详见[打包发布](#-打包发布)。

### 方式二：源码运行

```bash
# 1. 安装依赖（建议使用虚拟环境）
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. 启动
python entry_point.py
```

> 4-bit 量化需要 `bitsandbytes`（Windows 需与 CUDA 版本匹配）；未安装时会自动回退到普通加载。

---

## 📖 使用指南

### 1. 导入模型

「模型管理」→「导入模型」，支持三种来源：

- **ModelScope**：输入仓库 ID，如 `Qwen/Qwen3.5-2B`（国内下载快）
- **HuggingFace**：输入仓库 ID，如 `Qwen/Qwen3-1.7B`
- **本地路径**：选择本地模型目录

导入后自动解析参数量、架构、文件大小；双击列表行可查看详情。

### 2. 多模态蒸馏实验

1. 「蒸馏实验」页选择**教师模型**（如 Qwen3.5-2B）和**学生模型**（如 Qwen3.5-0.8B）
2. 「图像数据集配置」选择图片目录（留空则为纯文本蒸馏），可配置**多个提示词**实现每图多数据增强
3. 配置生成参数（温度/Top-P/Top-K/Beam/思考模式）、训练参数（LoRA/优化器/学习率等）、评估问题与可选参考答案
4. 点击「开始蒸馏」，右侧实时显示阶段进度、识别结果预览、速度与 ETA
5. 完成后自动生成评估报告，可在「实验监控」查看逐图识别详情并打开报告

### 3. 结构化剪枝实验

1. 「剪枝实验」选择目标模型
2. 选择策略（FFN 维度 / 注意力头 / 整层）、剪枝比例、重要性指标（L1/L2）
3. 可选 LoRA 恢复微调（epochs、学习率、LoRA 参数）
4. 点击「开始剪枝」——重建后的模型参数与文件**真实缩小**，评估报告对比剪枝前后

### 4. 实验对比

「实验对比」勾选多个实验，即可看到：

- 训练 Loss 叠加曲线
- **教师 / 蒸馏前 / 蒸馏后**三系列能力雷达图
- 维度评分表（完整性 / 覆盖率 / 流畅度 / 一致性 / 图像对齐）
- 逐题回答对比（多模态实验含图片缩略图列）

### 5. 实验模板与批量运行

- 配置好参数后点「保存」存为模板（存于 `data/templates/`）
- 点「批量」选择多个模板与多个学生模型，**按 模板×学生 顺序自动运行**，结束后生成 `batch_summary_*.md` 汇总报告
- 参数配置会自动记忆，重启应用后恢复

---

## ⚙️ 参数参考

### 数据生成（教师）

| 参数 | 默认 | 说明 |
|------|------|------|
| 生成数量 | 50 | 纯文本蒸馏时教师生成的数据条数 |
| 温度 / Top-P / Top-K | 0.7 / 0.9 / 0 | 采样参数（Top-K=0 不限制） |
| 重复惩罚 | 1.0 | 避免生成重复 |
| Beam 数 | 1 | >1 时启用束搜索（自动关闭采样） |
| 最大 Token | 128 | 教师回答最大长度 |
| 思考模式 | 开 | Qwen enable_thinking |
| 图片提示词 | - | 每行一个，多行 = 每张图生成多条数据 |
| 最大图片数 | 50 | 多模态蒸馏使用的图片数量 |

### 训练（学生）

| 参数 | 默认 | 说明 |
|------|------|------|
| LoRA Rank / Alpha / Dropout | 16 / 32 / 0.05 | 微调容量与正则 |
| 训练轮数 | 3 | 视觉任务 1~3 轮即可 |
| 学习率 | 5e-5 | 视觉-语言任务建议 1e-5~2e-5 |
| 权重衰减 | 0.01 | 正则化 |
| Warmup 步数 | 0 | 线性预热 |
| 梯度累积 | 1 | 等效增大批大小 |
| 优化器 | AdamW | AdamW / AdamW8bit / SGD |
| 批大小 / 最大序列长度 / 梯度裁剪 | 1 / 256 / 1.0 | 训练基本参数 |
| 随机种子 | 42 | 可复现 |
| 4-bit 量化 | 开 | nf4 / fp4，双量化，fp16/bf16 |

### 评估

| 参数 | 默认 | 说明 |
|------|------|------|
| 评估问题 | 3 个默认 | 每行一个 |
| 参考答案 | - | 每行一条，用于内容覆盖率自动评分 |
| 评估最大 Token | 64 | 评估回答长度 |
| CLIP 图像对齐 | 关 | 可选，首次需下载 CLIP 模型 |

### 剪枝

| 参数 | 默认 | 说明 |
|------|------|------|
| 策略 | FFN 维度 | FFN / 注意力头 / 整层 |
| 剪枝比例 | 20% | 建议 10%~20% 起步 |
| 重要性指标 | L1 | L1 / L2 范数 |
| 恢复微调 | 开 | LoRA 恢复，epochs / lr / LoRA 参数可调 |

---

## 🧠 评估体系

实验完成后自动计算四个维度（0~100）：

1. **回答完整性**：长度是否充分（相对参考答案或 80 字符目标）
2. **内容覆盖率**：与参考答案的匹配度（字符/词级 F1 + BLEU-1 + ROUGE-L 三合一；无参考答案时为 0）
3. **语言流畅度**：词型-词次比
4. **基线一致性**：与蒸馏/剪枝前回答的相似度

多模态实验可选开启 **CLIP 图像对齐评分**（描述与图片的语义匹配度，CLIPScore 思路）。

每次实验输出 `data/experiments/<实验名>/`：

```text
├─ model.safetensors        # 蒸馏/剪枝后的学生模型
├─ tokenizer/processor      # 分词器/处理器
├─ distillation_data.json   # 教师生成的蒸馏数据
└─ evaluation_report.md     # 评估报告（含逐图识别详情）
```

---

## 🗂️ 目录结构

```text
distillation_lab/
├─ app/
│  ├─ components/      # UI 组件（图表、GPU 仪表、折叠分组、导入对话框）
│  ├─ engines/         # 核心引擎：蒸馏、剪枝、评估
│  ├─ models/          # 数据层：SQLite、模型管理、模板存储
│  ├─ pages/           # 页面：首页/模型管理/蒸馏/剪枝/监控/对比
│  ├─ utils/           # 配置、主题、日志、GPU 监控、异常
│  ├─ workers/         # 后台线程：单实验 / 批量队列
│  ├─ main.py          # 应用入口
│  └─ main_window.py   # 主窗口
├─ data/               # 运行时数据（不入库）：db/ experiments/ logs/ templates/
├─ dist/               # 构建产物（不入库）：安装包 / 免安装版 / 便携版
├─ docs/research_notes.md   # 多模态蒸馏研究资料
├─ scripts/            # 打包与工具脚本
├─ tests/              # 单元测试
├─ build_exe.py        # 全量免安装版打包
├─ build_installer.py  # 安装包构建入口（scripts/build_installer.py）
├─ installer.py        # 安装程序（GUI）
├─ launcher.py         # 便携版一键启动器
└─ entry_point.py      # 入口（含启动错误捕获）
```

---

## 📦 打包发布

```bash
# 1. 全量免安装版（内置全部依赖，无需 Python）
python build_exe.py                 # 输出 dist/DistillationLab/（约 3.4GB）

# 2. 安装包（安装程序 + payload）
python scripts/build_installer.py   # 输出 dist/DistillationLab-安装包/

# 3. 便携版启动器（自动安装依赖）
pyinstaller --onefile --console --name DistillationLab --distpath dist launcher.py
python scripts/make_dist.py         # 组装 dist/DistillationLab便携版/
```

## 🧪 测试

```bash
python -m unittest discover -s tests
```

覆盖：评估评分（相似度/BLEU/ROUGE/报告）、剪枝重建（FFN/注意力/层）、模板存储、worker 持久化与批量汇总。

---

## ❓ 常见问题

**Q：没有 NVIDIA 显卡能用吗？**
能。应用自动以 CPU 模式运行（速度较慢）；4-bit 量化需 GPU + bitsandbytes，会自动降级。

**Q：导入模型时 ModelScope 报错？**
检查网络；或改用本地路径 / HuggingFace 导入。ModelScope 下载失败时可重试或设置镜像。

**Q：CLIP 图像对齐评分为 0？**
该选项默认关闭；开启后首次使用需联网下载 `openai/clip-vit-base-patch32`（约 600MB），且评估结果需含 `image_path`（新实验自动附带）。

**Q：应用启动慢？**
首次启动需加载 torch 等库（约 10~20 秒），之后正常；GPU 状态查询在后台线程完成，不阻塞界面。

**Q：剪枝后参数没变小？**
请确认使用"结构化剪枝"策略（FFN/注意力头/层均真实重建）；若模型架构不兼容会自动回退为置零（日志会提示）。

**Q：安装包报"缺少 payload.zip"？**
请把整个 `DistillationLab-安装包` 文件夹一起复制（`安装程序.exe` 与 `payload.zip` 必须同目录）。

---

## 🛠️ 技术栈

| 层 | 技术 |
|----|------|
| 桌面框架 | PyQt6 |
| 图表 | pyqtgraph |
| 核心计算 | PyTorch + Transformers |
| 微调 | PEFT (LoRA) |
| 数据存储 | SQLite + JSON + Safetensors |
| GPU 监控 | nvidia-smi / torch |
| 模型来源 | HuggingFace / ModelScope / 本地 |

## 📄 许可证

MIT License（本项目未随附正式 LICENSE 文件，如需商用请自行补充）。研究用途请遵循各基础模型（如 Qwen）的许可协议。
