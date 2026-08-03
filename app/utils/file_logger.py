"""文件日志系统 - 独立于 UI 的持久化日志，用于崩溃后排查"""
import logging
import os
import sys
import datetime
import traceback
from pathlib import Path
from ..utils.config import DATA_DIR


# 日志目录
LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 当前实验日志文件路径（由 setup_experiment_logger 设置）
_current_log_path = None


def _log_file_path(exp_id=None) -> Path:
    """生成日志文件路径"""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if exp_id:
        return LOG_DIR / f"exp_{exp_id}_{ts}.log"
    return LOG_DIR / f"app_{ts}.log"


def setup_experiment_logger(exp_id, log_dir=None):
    """为单个实验创建独立的文件日志

    返回 (logger, log_path) 供 worker 使用
    """
    global _current_log_path
    log_path = _log_file_path(exp_id)
    _current_log_path = log_path

    logger = logging.getLogger(f"experiment_{exp_id}")
    logger.setLevel(logging.DEBUG)

    # 避免重复添加 handler
    if not logger.handlers:
        handler = logging.FileHandler(str(log_path), encoding="utf-8", mode="w")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # 写入初始信息
    logger.info("=" * 60)
    logger.info(f"实验 #{exp_id} 日志开始")
    logger.info(f"Python: {sys.version}")
    logger.info(f"设备: {'CUDA' if _has_cuda() else 'CPU'}")
    logger.info("=" * 60)

    return logger, str(log_path)


def get_current_log_path():
    """获取当前日志文件路径（崩溃后可读）"""
    return str(_current_log_path) if _current_log_path else ""


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def log_cuda_memory(logger, tag=""):
    """记录当前 CUDA 显存状态（用于诊断 OOM）"""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"[显存] {tag} - allocated={allocated:.2f}GB reserved={reserved:.2f}GB")
    except Exception:
        pass


def write_crash_report(error_msg: str):
    """进程崩溃时写入最后的错误报告（供下次启动时查看）"""
    crash_path = LOG_DIR / "last_crash.log"
    try:
        with open(str(crash_path), "w", encoding="utf-8") as f:
            f.write(f"崩溃时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"最后日志: {_current_log_path}\n")
            f.write(f"错误信息: {error_msg}\n")
            f.write(f"堆栈:\n{traceback.format_exc()}\n")
        print(f"[崩溃报告] 已写入 {crash_path}")
    except Exception:
        pass


def setup_global_excepthook():
    """设置全局异常钩子，捕获所有未处理的异常并写入文件"""
    original_hook = sys.excepthook

    def global_hook(exc_type, exc_value, exc_tb):
        error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        write_crash_report(error_msg)
        # 仍然调用原始钩子（让 PyQt 也能处理）
        if original_hook is not sys.__excepthook__:
            original_hook(exc_type, exc_value, exc_tb)
        else:
            sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = global_hook
