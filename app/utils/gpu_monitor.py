"""GPU 状态监控 - 通过 nvidia-smi / torch 获取详细 GPU 信息

自动识别：名称、显存总量/已用/空闲、利用率、温度、功耗、
驱动版本、CUDA 运行时版本、计算能力（Compute Capability）。
"""
import subprocess
import re
import os
import threading
import time
import torch
from typing import Optional


_GPU_CACHE = {"ts": 0.0, "data": None}
_CACHE_TTL = 2.0
_poller_thread = None


def get_gpu_info(use_cache=True):
    """获取 GPU 状态信息，返回列表或 None

    每项字段：index, name, memory_total, memory_used, memory_free,
    utilization, temperature, power_w, driver_version, compute_cap
    """
    # 结果缓存：侧边栏仪表与详情对话框共享一次 nvidia-smi 查询，
    # 避免每 2 秒重复拉起子进程（既卡 UI 又会闪现 cmd 窗口）
    now = time.time()
    if use_cache and _GPU_CACHE["data"] is not None and now - _GPU_CACHE["ts"] < _CACHE_TTL:
        return _GPU_CACHE["data"]

    result_data = _query_gpu_info()
    if use_cache:
        _GPU_CACHE["ts"] = time.time()
        _GPU_CACHE["data"] = result_data
    return result_data


def _query_gpu_info():
    """实际查询 GPU（无缓存）"""
    if not torch.cuda.is_available():
        return None

    try:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used,memory.free,"
             "utilization.gpu,temperature.gpu,power.draw,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=creationflags,
        )
        if result.returncode != 0:
            return _fallback_gpu_info()

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 10:
                # nvidia-smi 返回 MiB，统一换算为 GB
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total": round(float(parts[2]) / 1024, 2),
                    "memory_used": round(float(parts[3]) / 1024, 2),
                    "memory_free": round(float(parts[4]) / 1024, 2),
                    "utilization": int(parts[5]),
                    "temperature": int(parts[6]),
                    "power_w": float(parts[7]),
                    "driver_version": parts[8],
                    "compute_cap": parts[9],
                })
        return gpus if gpus else _fallback_gpu_info()
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
        return _fallback_gpu_info()


def _fallback_gpu_info():
    """备选方案：直接从 torch 获取 GPU 信息"""
    if not torch.cuda.is_available():
        return None
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_total = props.total_memory / (1024 ** 3)  # GB
        mem_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        try:
            mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        except AttributeError:
            mem_reserved = mem_allocated
        gpus.append({
            "index": i,
            "name": props.name,
            "memory_total": round(mem_total, 1),
            "memory_used": round(mem_reserved, 1),
            "memory_free": round(max(0, mem_total - mem_reserved), 1),
            "utilization": 0,
            "temperature": 0,
            "power_w": 0.0,
            "driver_version": "unknown",
            "compute_cap": ".".join(str(x) for x in torch.cuda.get_device_capability(i)),
        })
    return gpus


def start_gpu_poller(interval: float = 2.0):
    """后台线程定时刷新 GPU 缓存，UI 线程只读缓存，避免卡顿与窗口闪现"""
    global _poller_thread
    if _poller_thread is not None and _poller_thread.is_alive():
        return

    def _loop():
        while True:
            try:
                data = _query_gpu_info()
                _GPU_CACHE["ts"] = time.time()
                _GPU_CACHE["data"] = data
            except Exception:
                pass
            time.sleep(interval)

    _poller_thread = threading.Thread(target=_loop, daemon=True, name="gpu-poller")
    _poller_thread.start()


def get_cuda_runtime_info() -> dict:
    """获取 CUDA 运行时信息（版本、库）"""
    info = {"torch": torch.__version__, "cuda": "N/A", "cudnn": "N/A"}
    try:
        info["cuda"] = torch.version.cuda or "N/A"
    except Exception:
        pass
    try:
        info["cudnn"] = torch.backends.cudnn.version() or "N/A"
    except Exception:
        pass
    return info


def format_gpu_info(gpu) -> str:
    """格式化 GPU 信息为字符串"""
    mem_pct = (gpu["memory_used"] / gpu["memory_total"] * 100) if gpu["memory_total"] > 0 else 0
    text = (f"GPU {gpu['index']}: {gpu['name']} | "
            f"显存: {gpu['memory_used']:.1f}/{gpu['memory_total']:.1f} GB ({mem_pct:.0f}%) | "
            f"利用率: {gpu['utilization']}%")
    if gpu.get("temperature"):
        text += f" | 温度: {gpu['temperature']}°C"
    if gpu.get("power_w"):
        text += f" | 功耗: {gpu['power_w']:.0f}W"
    return text


def format_gpu_detail(gpu) -> str:
    """格式化 GPU 详细信息（多行）"""
    lines = [
        f"GPU {gpu['index']}: {gpu['name']}",
        f"  驱动版本: {gpu.get('driver_version', 'unknown')}",
        f"  计算能力: {gpu.get('compute_cap', 'unknown')}",
        f"  显存: 已用 {gpu['memory_used']:.1f} GB / 空闲 {gpu.get('memory_free', 0):.1f} GB / 总量 {gpu['memory_total']:.1f} GB",
        f"  利用率: {gpu.get('utilization', 0)}%",
    ]
    if gpu.get("temperature"):
        lines.append(f"  温度: {gpu['temperature']}°C")
    if gpu.get("power_w"):
        lines.append(f"  功耗: {gpu['power_w']:.0f} W")
    return "\n".join(lines)
