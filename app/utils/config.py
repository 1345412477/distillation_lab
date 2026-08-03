"""全局配置管理"""
import os
import sys
from pathlib import Path

# 项目根目录
if getattr(sys, 'frozen', False):
    # PyInstaller 打包模式下，数据目录放在 exe 同级
    ROOT_DIR = Path(sys.executable).parent.absolute()
else:
    # 源码开发模式
    ROOT_DIR = Path(__file__).parent.parent.parent.absolute()

# 数据目录
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
EXPERIMENT_DIR = DATA_DIR / "experiments"
DB_DIR = DATA_DIR / "db"
DB_PATH = DB_DIR / "distillation_lab.db"

# 创建目录
for d in [DATA_DIR, MODEL_DIR, EXPERIMENT_DIR, DB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 默认种子问题集
SEED_QUESTIONS = [
    "什么是深度学习？请用简单的话解释。",
    "解释一下什么是过拟合，以及如何避免它。",
    "HTTP和HTTPS有什么区别？",
    "什么是RESTful API？",
    "解释什么是梯度下降算法。",
    "用Python写一个计算斐波那契数列的函数。",
    "如何用Python读取一个CSV文件？",
    "写一个Python函数来判断字符串是否是回文。",
    "如何用Python实现二分查找？",
    "写一段Python代码来统计列表中每个元素出现的次数。",
    "一个房间里有3盏灯，门外有3个开关，每个开关控制一盏灯。你只能进房间一次，如何确定每个开关控制哪盏灯？",
    "如果所有的猫都是动物，所有的动物都会动，那么所有的猫都会动吗？为什么？",
    "小明有5个苹果，给了小红2个，又买了3个，他现在有几个苹果？",
    "一列火车从A站到B站需要2小时，如果速度提高一倍，需要多长时间？",
    "一个水池有进水管和出水管，进水管每小时注水3升，出水管每小时排水2升，水池容量为30升，从空池开始，多久能注满？",
]

# 默认评估问题集
EVAL_QUESTIONS = [
    "什么是深度学习？",
    "用Python写一个计算斐波那契数列的函数。",
    "HTTP和HTTPS有什么区别？",
]

# 多模态评估问题默认集
IMAGE_EVAL_QUESTIONS = [
    "请描述这张图片中的主要内容",
    "这张图片中有什么物体?",
    "请描述图片中的场景和氛围",
]

# 蒸馏默认参数
DEFAULT_DISTILL_CONFIG = {
    "num_generate": 50,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "num_beams": 1,
    "max_new_tokens": 128,
    "enable_think": True,
    "train_method": "lora",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "epochs": 3,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_steps": 0,
    "grad_accum_steps": 1,
    "optimizer": "adamw",
    "batch_size": 1,
    "max_seq_len": 256,
    "grad_clip": 1.0,
    "seed": 42,
    "use_4bit": True,
    "quant_type": "nf4",
    "double_quant": True,
    "compute_dtype": "fp16",
    "eval_max_new_tokens": 64,
}

# 剪枝默认参数
DEFAULT_PRUNE_CONFIG = {
    "prune_strategy": "ffn",
    "prune_ratio": 0.2,
    "importance_metric": "l1",
    "prune_mode": "zero",
    "enable_recovery": True,
    "recovery_method": "lora",
    "recovery_data_source": "distill_data",
    "recovery_epochs": 5,
    "recovery_lr": 5e-5,
    "recovery_lora_r": 16,
    "recovery_lora_alpha": 32,
    "recovery_lora_dropout": 0.05,
    "seed": 42,
    "eval_max_new_tokens": 64,
}

# 多模态蒸馏默认配置
DEFAULT_IMAGE_CONFIG = {
    "image_dir": "",
    "image_extensions": [".jpg", ".jpeg", ".png", ".webp"],
    "image_prompt_template": "请详细描述这张图片的内容",
    "max_images": 50,
    "eval_image_prompt": "请描述这张图片中的主要内容",
    "image_prompts": ["请详细描述这张图片的内容"],
}

# 模型来源
MODEL_SOURCE_CHOICES = ["huggingface", "modelscope", "local"]
