"""DistillationLab 一键启动器

在目标电脑上双击本程序（或 DistillationLab.exe）即可：
1. 检测系统 Python（py 启动器 / python / python3）
2. 自动创建虚拟环境 app_env
3. 自动安装 requirements.txt 中的全部依赖（首次约 1~2GB，请耐心等待）
4. 启动图形界面（pythonw，无控制台窗口）

构建为独立 exe 的命令：
    pyinstaller --onefile --console --name DistillationLab --distpath dist launcher.py
"""
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request


APP_ROOT = pathlib.Path(
    sys.executable if getattr(sys, "frozen", False) else __file__
).resolve().parent
VENV_DIR = APP_ROOT / "app_env"
REQ_FILE = APP_ROOT / "requirements.txt"
MARKER = VENV_DIR / ".deps_ok"
PYTHON = VENV_DIR / "Scripts" / "python.exe"
PYTHONW = VENV_DIR / "Scripts" / "pythonw.exe"
AUTO_PYTHON_VERSION = "3.11.9"
AUTO_PYTHON_DIR = APP_ROOT / "runtime_py"
PYTHON_INSTALLER_URL = (
    f"https://www.python.org/ftp/python/{AUTO_PYTHON_VERSION}/"
    f"python-{AUTO_PYTHON_VERSION}-amd64.exe"
)
PYTHON_MIRRORS = [
    PYTHON_INSTALLER_URL,
    f"https://mirrors.huaweicloud.com/python/{AUTO_PYTHON_VERSION}/"
    f"python-{AUTO_PYTHON_VERSION}-amd64.exe",
    f"https://registry.npmmirror.com/-/binary/python/{AUTO_PYTHON_VERSION}/"
    f"python-{AUTO_PYTHON_VERSION}-amd64.exe",
]


def log(msg: str):
    print(f"[DistillationLab] {msg}", flush=True)


def _pause():
    """等待用户按键；stdout 被重定向时改为等待数秒"""
    try:
        input("\n按回车退出...")
    except EOFError:
        time.sleep(8)


def download_file(urls, dest, desc: str = ""):
    """下载文件（优先 curl，多镜像自动切换，失败自动重试）"""
    if isinstance(urls, str):
        urls = [urls]
    log(f"正在下载 {desc}...")
    curl = shutil.which("curl")
    last_err = None
    for url in urls:
        try:
            if curl:
                r = subprocess.run(
                    [curl, "-L", "--fail", "--silent", "--show-error",
                     "--output", str(dest), url],
                    timeout=600,
                )
                if r.returncode == 0 and dest.is_file() and dest.stat().st_size > 1000:
                    log(f"下载完成: {desc}（{dest.stat().st_size / 1e6:.1f} MB）")
                    return dest
                last_err = RuntimeError(f"curl 退出码 {r.returncode}")
            else:
                tmp = dest.with_suffix(".part")
                with urllib.request.urlopen(url, timeout=120) as resp:
                    with open(tmp, "wb") as f:
                        while True:
                            chunk = resp.read(1 << 16)
                            if not chunk:
                                break
                            f.write(chunk)
                tmp.replace(dest)
                if dest.stat().st_size > 1000:
                    log(f"下载完成: {desc}（{dest.stat().st_size / 1e6:.1f} MB）")
                    return dest
                last_err = RuntimeError("下载文件过小")
        except Exception as e:
            last_err = e
            log(f"  镜像下载失败: {url[:70]}... ({e})")
        if dest.exists():
            dest.unlink(missing_ok=True)
    raise RuntimeError(f"所有镜像下载失败: {last_err}")


def auto_install_python() -> str:
    """无系统 Python 时：下载官方 Python 并静默安装到应用目录（仅当前用户，免管理员）"""
    installer = APP_ROOT / f"python-{AUTO_PYTHON_VERSION}-amd64.exe"
    if not installer.is_file():
        download_file(PYTHON_MIRRORS, installer,
                      f"Python {AUTO_PYTHON_VERSION} 安装程序（约 28MB）")
    log("正在静默安装 Python（仅当前用户，无需管理员，请稍候 1~3 分钟）...")
    subprocess.run(
        [str(installer), "/quiet",
         "InstallAllUsers=0", "PrependPath=0", "Include_launcher=0",
         "Include_test=0", "Include_doc=0", f"TargetDir={AUTO_PYTHON_DIR}"],
        check=True,
    )
    py = AUTO_PYTHON_DIR / "python.exe"
    if not py.is_file():
        raise RuntimeError("Python 自动安装失败，请手动安装 Python 3.10+ 后重试")
    log(f"Python 已自动安装: {py}")
    return str(py)


def _try_run(cmd) -> str:
    """尝试执行命令并返回首行输出；失败返回空串"""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    except Exception:
        return ""


def find_system_python():
    """寻找可用于创建虚拟环境的系统 Python"""
    for minor in ("13", "12", "11", "10"):
        path = _try_run(["py", f"-3.{minor}", "-c", "import sys; print(sys.executable)"])
        if path and os.path.isfile(path):
            return path
    path = _try_run(["python", "-c", "import sys; print(sys.executable)"])
    if path and os.path.isfile(path):
        return path
    path = _try_run(["python3", "-c", "import sys; print(sys.executable)"])
    if path and os.path.isfile(path):
        return path
    return None


def req_hash() -> str:
    if REQ_FILE.exists():
        return hashlib.md5(REQ_FILE.read_bytes()).hexdigest()
    return "none"


def create_venv(sys_py: str):
    log(f"创建虚拟环境: {VENV_DIR}")
    subprocess.run([sys_py, "-m", "venv", str(VENV_DIR)], check=True)
    subprocess.run([str(PYTHON), "-m", "pip", "install", "--upgrade", "pip"],
                   check=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)


def install_deps():
    log("正在安装依赖（首次约 1~2GB，视网速可能需要 5~30 分钟）...")
    log("国内网络较慢时，可设置环境变量 DISTILL_MIRROR=1 使用清华镜像")
    cmd = [str(PYTHON), "-m", "pip", "install", "-r", str(REQ_FILE)]
    if os_environ_mirror():
        cmd += ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    for attempt in (1, 2, 3):
        try:
            subprocess.run(cmd, check=True, creationflags=flags)
            MARKER.write_text(req_hash(), encoding="utf-8")
            return
        except subprocess.CalledProcessError:
            log(f"依赖安装失败（第 {attempt}/3 次），5 秒后自动重试...")
            time.sleep(5)
    raise RuntimeError("依赖安装连续失败，请检查网络后重试（可设置 DISTILL_MIRROR=1 使用国内镜像）")


def os_environ_mirror() -> bool:
    import os
    return os.environ.get("DISTILL_MIRROR", "").strip() in ("1", "true", "yes")


def launch():
    log("启动 DistillationLab 图形界面...")
    subprocess.Popen(
        [str(PYTHONW), str(APP_ROOT / "entry_point.py")],
        cwd=str(APP_ROOT),
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )


def main() -> int:
    log(f"应用目录: {APP_ROOT}")

    # 环境损坏时重建
    if VENV_DIR.exists() and not PYTHON.exists():
        log("检测到环境损坏，正在重建...")
        shutil.rmtree(VENV_DIR, ignore_errors=True)

    if not PYTHON.exists():
        sys_py = find_system_python()
        if not sys_py:
            # 未找到系统 Python：自动下载并配置（交互时征求同意）
            proceed = True
            try:
                answer = input(
                    "未找到 Python。是否自动下载并配置 Python 3.11？(Y/n) "
                ).strip().lower()
                proceed = answer in ("", "y", "yes")
            except EOFError:
                proceed = True  # 非交互环境默认自动配置
            if not proceed:
                log("已取消。可手动安装 Python 3.10+ 后重试。")
                _pause()
                return 1
            if AUTO_PYTHON_DIR.joinpath("python.exe").is_file():
                sys_py = str(AUTO_PYTHON_DIR / "python.exe")
            else:
                sys_py = auto_install_python()
        create_venv(sys_py)

    # 依赖未安装或 requirements 有变化时自动安装
    if not MARKER.exists() or MARKER.read_text(encoding="utf-8").strip() != req_hash():
        install_deps()
    else:
        log("依赖已就绪，跳过安装")

    launch()
    log("已启动。如无窗口弹出，请检查 entry_point.py 与 app/ 目录是否完整。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[DistillationLab] 启动失败: {e}", file=sys.stderr, flush=True)
        _pause()
        sys.exit(1)
