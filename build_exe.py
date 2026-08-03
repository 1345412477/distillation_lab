"""
DistillationLab packaging script
Packages the application into a single-directory exe using PyInstaller

Usage: python build_exe.py
Output: dist/DistillationLab/
"""
import PyInstaller.__main__
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "dist")

opts = [
    # Entry point script (with error capture)
    os.path.join(ROOT, "entry_point.py"),

    # Output
    "--distpath", OUTPUT_DIR,
    "--name=DistillationLab",

    # Mode
    "--onedir",
    "--noconsole",         # No console window (pure GUI app)
    "--clean",
    "--noconfirm",

    # Search path
    "--paths", ROOT,

    # ===== Hidden imports =====
    "--hidden-import=PyQt6.QtCore",
    "--hidden-import=PyQt6.QtWidgets",
    "--hidden-import=PyQt6.QtGui",
    "--hidden-import=PyQt6.sip",
    "--hidden-import=pyqtgraph",

    # ===== 动态导入大户：整包收集，避免运行时缺模块 =====
    "--collect-all=torch",
    "--collect-all=transformers",
    "--collect-all=modelscope",
    "--collect-all=peft",
    "--collect-all=accelerate",
    "--collect-all=bitsandbytes",
    "--collect-all=huggingface_hub",
    "--collect-all=tokenizers",
    "--collect-all=safetensors",

    # 项目代码
    "--collect-all=app",

    # Exclude unnecessary packages (reduce size)
    "--exclude-module=torchvision",
    "--exclude-module=torchaudio",
    "--exclude-module=matplotlib",
    "--exclude-module=scipy",
    "--exclude-module=tkinter",
    "--exclude-module=tensorflow",
    "--exclude-module=torch_tb_profiler",
]

# ============================================================
# Execute packaging
# ============================================================
print("=" * 60)
print("DistillationLab Packaging Tool")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"Project: {ROOT}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 60)
print()
print("Packaging in progress, please wait... (may take 5-15 minutes)")
print()

PyInstaller.__main__.run(opts)

print()
print("=" * 60)
print("Packaging complete!")
exe_path = os.path.join(OUTPUT_DIR, "DistillationLab", "DistillationLab.exe")
print(f"Launch file: {exe_path}")
print()

# Copy debug script to output directory
import shutil
bat_src = os.path.join(ROOT, "DistillationLab_debug.bat")
bat_dst = os.path.join(OUTPUT_DIR, "DistillationLab", "DistillationLab_debug.bat")
shutil.copy2(bat_src, bat_dst)
print(f"Debug script: {bat_dst}")
print()
print("For debugging, use DistillationLab_debug.bat")
print("=" * 60)
