"""模型管理器 - 处理模型加载、配置解析等"""
import json
import os
from pathlib import Path
from ..utils.config import MODEL_DIR


def _find_config_json(local_path: str) -> str | None:
    """在模型目录中查找 config.json，支持嵌套子目录"""
    # 优先直接查找
    direct = os.path.join(local_path, "config.json")
    if os.path.exists(direct):
        return direct
    # ModelScope 下载后文件在 snapshots/<branch>/ 下
    snapshots_dir = os.path.join(local_path, "snapshots")
    if os.path.isdir(snapshots_dir):
        for entry in os.listdir(snapshots_dir):
            candidate = os.path.join(snapshots_dir, entry, "config.json")
            if os.path.exists(candidate):
                return candidate
    return None


def parse_model_config(local_path: str) -> dict:
    """解析模型目录中的 config.json，返回模型信息"""
    config_path = _find_config_json(local_path)
    if not config_path:
        return {"architecture": "unknown", "parameters": 0}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"architecture": "unknown", "parameters": 0}

    info = {}

    # 架构名称
    architectures = cfg.get("architectures", [])
    info["architecture"] = architectures[0] if architectures else cfg.get("model_type", "unknown")

    # 支持嵌套配置（如 Qwen3.5 的参数在 text_config 内）
    text_config = cfg.get("text_config", {})
    effective_cfg = text_config if text_config else cfg

    # 参数量估算
    hidden_size = effective_cfg.get("hidden_size", 0)
    num_layers = effective_cfg.get("num_hidden_layers", effective_cfg.get("num_layers", 0))
    intermediate_size = effective_cfg.get("intermediate_size", hidden_size * 4)
    vocab_size = effective_cfg.get("vocab_size", 0)

    info["hidden_size"] = hidden_size
    info["num_layers"] = num_layers
    info["vocab_size"] = vocab_size
    info["dtype"] = effective_cfg.get("dtype", cfg.get("dtype", "float32"))

    # 估算参数量
    if hidden_size > 0 and num_layers > 0:
        # 简化参数估算：每层 attention + FFN
        num_heads = effective_cfg.get("num_attention_heads", 1)
        num_kv_heads = effective_cfg.get("num_key_value_heads", num_heads)
        head_dim = effective_cfg.get("head_dim", hidden_size // num_heads if num_heads > 0 else 0)

        # Embedding
        params = vocab_size * hidden_size * 2  # 输入 + 输出(如果有tie)
        if effective_cfg.get("tie_word_embeddings", True):
            params = vocab_size * hidden_size

        # 每层
        for _ in range(num_layers):
            # QKV
            params += hidden_size * num_heads * head_dim  # q
            params += hidden_size * num_kv_heads * head_dim  # k
            params += hidden_size * num_kv_heads * head_dim  # v
            params += hidden_size * hidden_size  # o
            # FFN
            params += hidden_size * intermediate_size  # gate
            params += hidden_size * intermediate_size  # up
            params += intermediate_size * hidden_size  # down
            # RMS norm x2
            params += hidden_size * 2

        # Final norm + lm_head
        params += hidden_size  # final norm
        if not effective_cfg.get("tie_word_embeddings", True):
            params += hidden_size * vocab_size

        info["parameters"] = round(params / 1e9, 3)  # B
    else:
        info["parameters"] = 0

    # 文件大小（支持嵌套子目录查找）
    safetensors_files = list(Path(local_path).rglob("*.safetensors"))
    bin_files = list(Path(local_path).rglob("*.bin"))
    model_files = safetensors_files or bin_files
    if model_files:
        total_size = sum(f.stat().st_size for f in model_files)
        info["file_size"] = f"{total_size / 1e9:.1f}GB"
    else:
        info["file_size"] = "unknown"

    return info


def get_model_path(repo_id: str, source: str = "modelscope") -> str:
    """从 ModelScope/HuggingFace 下载模型，返回本地路径"""
    if source == "modelscope":
        from modelscope import snapshot_download
        return snapshot_download(repo_id)
    elif source == "huggingface":
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id)
    elif source == "local":
        return repo_id
    return repo_id


def resolve_model_path(local_path: str) -> str:
    """将 ModelScope 的缓存根目录解析为实际 snapshot 路径

    ModelScope snapshot_download 返回的是缓存根目录（如
    ~/.cache/modelscope/models/Qwen--Qwen3.5-0.8B），而 transformer
    的 from_pretrained 需要指向包含 config.json 的具体 snapshot 子目录。

    此函数自动检测并返回实际 snapshot 目录，如果已经是有效路径则原样返回。
    """
    if not local_path:
        return local_path

    # 已经是 snapshot 子目录（包含 config.json）或本地非 modelscope 路径
    config_path = os.path.join(local_path, "config.json")
    if os.path.isfile(config_path):
        return local_path

    # 查找 snapshots 子目录下的第一个分支
    snapshots_dir = os.path.join(local_path, "snapshots")
    if os.path.isdir(snapshots_dir):
        for entry in os.listdir(snapshots_dir):
            candidate = os.path.join(snapshots_dir, entry)
            if os.path.isdir(candidate):
                # 验证该目录包含 config.json
                if os.path.isfile(os.path.join(candidate, "config.json")):
                    return candidate

    # 无法解析，返回原路径
    return local_path


def get_available_model_ids(source: str = "modelscope") -> list:
    """返回预设的推荐模型列表"""
    if source == "modelscope":
        return [
            "Qwen/Qwen3-1.7B",
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen2.5-1.5B",
        ]
    elif source == "huggingface":
        return [
            "Qwen/Qwen3-1.7B",
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen2.5-0.5B",
        ]
    return []
