"""组装便携版分发目录

用法: python scripts/make_dist.py
输出: dist/DistillationLab便携版/
包含: DistillationLab.exe（启动器）+ app/ + entry_point.py + requirements.txt
目标电脑双击 DistillationLab.exe 即可自动安装依赖并启动。
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist"
OUT_DIR = DIST_DIR / "DistillationLab便携版"

COPY_ITEMS = [
    "app",
    "entry_point.py",
    "requirements.txt",
    "README.md",
    "启动应用.bat",
    "launcher.py",
]

EXE_SOURCE = DIST_DIR / "DistillationLab.exe"


def main():
    if not EXE_SOURCE.exists():
        print(f"[错误] 未找到 {EXE_SOURCE}，请先执行:")
        print("  pyinstaller --onefile --console --name DistillationLab --distpath dist launcher.py")
        return 1

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    for item in COPY_ITEMS:
        src = ROOT / item
        dst = OUT_DIR / item
        if src.is_dir():
            shutil.copytree(
                src, dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"),
            )
        elif src.is_file():
            shutil.copy2(src, dst)

    shutil.copy2(EXE_SOURCE, OUT_DIR / "DistillationLab.exe")

    # 空 data 目录（首次启动自动建表/日志）
    (OUT_DIR / "data" / "db").mkdir(parents=True)
    (OUT_DIR / "data" / "logs").mkdir(parents=True)

    readme = OUT_DIR / "使用说明.txt"
    readme.write_text(
        "DistillationLab 便携版\n"
        "======================\n"
        "1. 双击 DistillationLab.exe\n"
        "2. 首次运行会自动创建虚拟环境并安装依赖（约 1~2GB，需联网）\n"
        "3. 安装完成后自动打开图形界面\n"
        "4. 依赖只需安装一次；requirements.txt 有更新时会自动补装\n\n"
        "要求: Windows 10/11；无需安装 Python——未检测到时启动器会\n"
        "      自动下载并静默安装 Python 3.11（仅当前用户）\n"
        "GPU 加速: 本机有 NVIDIA 显卡时自动使用；无显卡时以 CPU 模式运行\n\n"
        "如需完全离线/免安装版本，请用 build_exe.py 做全量打包（体积很大）。\n",
        encoding="utf-8",
    )
    print(f"便携版已生成: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
