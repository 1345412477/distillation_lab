"""构建 DistillationLab 安装包

用法: python scripts/build_installer.py
输出: dist/DistillationLab-安装包/
  ├─ 安装程序.exe      （GUI 安装程序，PyQt6）
  ├─ _internal/        （安装程序运行库）
  ├─ payload.zip       （全量应用，约 3.4GB）
  └─ 安装说明.txt
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
APP_DIR = DIST / "DistillationLab"
PACK_DIR = DIST / "DistillationLab-安装包"


def make_payload_zip():
    """把全量应用压缩为 payload.zip"""
    if not APP_DIR.is_dir():
        raise FileNotFoundError(f"未找到全量应用: {APP_DIR}，请先运行 build_exe.py")
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = PACK_DIR / "payload.zip"
    if zip_path.is_file():
        print(f"payload.zip 已存在（{zip_path.stat().st_size / 1e9:.2f} GB），跳过压缩")
        return
    print(f"压缩应用 -> {zip_path}（约 3.4GB，需要几分钟）...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        files = [p for p in APP_DIR.rglob("*") if p.is_file()]
        for i, f in enumerate(files):
            zf.write(f, f.relative_to(APP_DIR).as_posix())
            if i % 500 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    print(f"payload.zip 完成: {zip_path.stat().st_size / 1e9:.2f} GB")


def build_installer_exe():
    """用 PyInstaller 构建安装程序（onefile，运行库内嵌，可单独分发）"""
    py = sys.executable
    work = ROOT / "build_installer"
    work.mkdir(exist_ok=True)
    # 清理旧的 onedir 残留（_internal 等）
    for old in ("_internal",):
        p = PACK_DIR / old
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    cmd = [
        py, "-m", "PyInstaller",
        "--onefile", "--noconsole", "--noconfirm", "--clean",
        "--name", "安装程序",
        "--distpath", str(PACK_DIR),
        "--workpath", str(work / "work"),
        "--specpath", str(work / "spec"),
        str(ROOT / "installer.py"),
    ]
    print("构建安装程序 exe...")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def write_readme():
    (PACK_DIR / "安装说明.txt").write_text(
        "DistillationLab 安装包\n"
        "======================\n"
        "重要：请把整个“DistillationLab-安装包”文件夹一起复制到目标电脑，\n"
        "      “安装程序.exe”必须与“payload.zip”在同一文件夹。\n\n"
        "1. 双击“安装程序.exe”\n"
        "2. 选择安装目录（默认 %LOCALAPPDATA%\\Programs\\DistillationLab）\n"
        "3. 点击“开始安装”，等待进度条完成（约 3.5GB，需要几分钟）\n"
        "4. 安装完成后可选择立即启动，桌面会生成快捷方式\n\n"
        "卸载：运行安装目录下的“卸载.bat”\n"
        "要求：Windows 10/11；无需安装 Python、无需联网（已内置全部依赖）\n"
        "GPU：有 NVIDIA 显卡时自动启用加速，否则以 CPU 模式运行\n",
        encoding="utf-8",
    )


def main():
    make_payload_zip()
    build_installer_exe()
    write_readme()
    size = sum(p.stat().st_size for p in PACK_DIR.rglob("*") if p.is_file())
    print(f"\n安装包已生成: {PACK_DIR}（{size / 1e9:.2f} GB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
