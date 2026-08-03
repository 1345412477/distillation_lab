"""DistillationLab 安装程序

将全量版应用（dist/DistillationLab）安装到用户指定目录：
- 选择安装目录、创建桌面快捷方式
- 进度条显示解压进度
- 安装目录内置 卸载.bat

构建:
    python scripts/build_installer.py

自动化测试（免界面）:
    DistillationLab-安装程序.exe --silent C:\目标目录
"""
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QLineEdit, QPushButton, QFileDialog,
                              QCheckBox, QProgressBar, QMessageBox)


APP_NAME = "DistillationLab"


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def payload_source() -> Path:
    """优先使用安装包同目录的 payload.zip，否则用源码 dist/DistillationLab"""
    root = app_root()
    zip_path = root / "payload.zip"
    if zip_path.is_file():
        return zip_path
    src = root.parent / "dist" / "DistillationLab"
    return src if src.is_dir() else None


def create_shortcut(exe_path: Path, working_dir: Path, desktop: bool = True):
    """创建桌面快捷方式（PowerShell + WScript.Shell）"""
    desktop_dir = Path(os.path.join(os.environ["USERPROFILE"], "Desktop"))
    lnk = (desktop_dir if desktop else working_dir) / f"{APP_NAME}.lnk"
    ps = (
        f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}');"
        f"$s.TargetPath='{exe_path}';"
        f"$s.WorkingDirectory='{working_dir}';"
        f"$s.Description='模型蒸馏与剪枝实验平台';"
        f"$s.Save()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        check=True,
    )
    return lnk


def write_uninstaller(target: Path):
    """写入卸载脚本（删除快捷方式与本目录）"""
    bat = target / "卸载.bat"
    bat.write_text(
        "@echo off\r\n"
        "chcp 65001 >nul\r\n"
        "title 卸载 DistillationLab\r\n"
        "echo 正在卸载 DistillationLab...\r\n"
        'del "%USERPROFILE%\\Desktop\\DistillationLab.lnk" >nul 2>nul\r\n'
        "ping 127.0.0.1 -n 2 >nul\r\n"
        'rd /s /q "%~dp0"\r\n'
        "echo 卸载完成\r\n"
        "pause\r\n",
        encoding="gbk",
    )


def install(src, target: Path, progress_cb=None) -> int:
    """安装应用；返回复制的文件数"""
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    if isinstance(src, Path) and src.suffix == ".zip":
        with zipfile.ZipFile(src) as zf:
            names = zf.namelist()
            total = len(names)
            for i, name in enumerate(names):
                zf.extract(name, target)
                count += 1
                if progress_cb and i % 50 == 0:
                    progress_cb(i + 1, total)
    else:
        src = Path(src)
        files = [p for p in src.rglob("*") if p.is_file()]
        total = len(files)
        for i, f in enumerate(files):
            rel = f.relative_to(src)
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            count += 1
            if progress_cb and i % 100 == 0:
                progress_cb(i + 1, total)
    write_uninstaller(target)
    return count


class InstallerWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"安装 {APP_NAME}")
        self.setMinimumWidth(520)
        self._payload = payload_source()
        self._init_ui()
        if self._payload is None:
            self.install_btn.setEnabled(False)
            QMessageBox.critical(
                self, "缺少安装数据",
                "未找到安装数据 payload.zip。\n\n"
                "请保持“安装程序.exe”与“payload.zip”在同一文件夹（即整个"
                "“DistillationLab-安装包”文件夹一起复制），不要只复制 exe。",
            )

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel(f"{APP_NAME} 安装程序")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        desc = QLabel(
            "模型蒸馏与剪枝实验平台（含 torch / transformers 等全部依赖，约 3.5GB）\n"
            "安装后无需联网、无需安装 Python，双击即可运行。"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addWidget(QLabel("安装目录:"))
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit(self._default_dir())
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        self.shortcut_cb = QCheckBox("创建桌面快捷方式")
        self.shortcut_cb.setChecked(True)
        layout.addWidget(self.shortcut_cb)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        self.install_btn = QPushButton("开始安装")
        self.install_btn.clicked.connect(self._do_install)
        cancel_btn = QPushButton("退出")
        cancel_btn.clicked.connect(self.close)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self.install_btn)
        layout.addLayout(btn_row)

    @staticmethod
    def _default_dir() -> str:
        return os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
                            "Programs", APP_NAME)

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "选择安装目录", self._default_dir())
        if path:
            self.path_edit.setText(path)

    def _do_install(self):
        target = Path(self.path_edit.text().strip())
        if not target.is_absolute():
            QMessageBox.warning(self, "提示", "请选择有效的安装目录")
            return
        self.install_btn.setEnabled(False)

        def cb(cur, total):
            self.progress.setRange(0, total)
            self.progress.setValue(cur)
            self.status_label.setText(f"正在复制... {cur}/{total}")
            QApplication.processEvents()

        try:
            count = install(self._payload, target, progress_cb=cb)
            self.progress.setValue(self.progress.maximum())
            self.status_label.setText(f"完成：{count} 个文件")

            exe_path = target / "DistillationLab.exe"
            if self.shortcut_cb.isChecked() and exe_path.exists():
                create_shortcut(exe_path, target)
                self.status_label.setText("完成：已创建桌面快捷方式")

            reply = QMessageBox.question(
                self, "安装完成",
                f"安装完成！\n{target}\n\n是否立即启动应用？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes and exe_path.exists():
                subprocess.Popen([str(exe_path)], cwd=str(target))
        except Exception as e:
            QMessageBox.critical(self, "安装失败", str(e))
        finally:
            self.install_btn.setEnabled(True)


def silent_install(target: str) -> int:
    """免界面安装（供自动化测试/静默部署）"""
    src = payload_source()
    if src is None:
        print("payload 缺失")
        return 1
    count = install(src, Path(target))
    exe = Path(target) / "DistillationLab.exe"
    if exe.exists():
        try:
            create_shortcut(exe, Path(target))
        except Exception:
            pass
    print(f"OK installed {count} files to {target}")
    return 0


def main():
    # 冻结模式下若缺少 payload，给出友好提示（避免静默失败）
    if getattr(sys, "frozen", False) and payload_source() is None:
        if "--silent" not in sys.argv:
            app = QApplication(sys.argv)
            w = InstallerWindow()
            w.show()
            return app.exec()
        print("缺少 payload.zip，安装中止。请保持 安装程序.exe 与 payload.zip 在同一文件夹。")
        return 1
    if "--silent" in sys.argv:
        idx = sys.argv.index("--silent")
        if idx + 1 < len(sys.argv):
            return silent_install(sys.argv[idx + 1])
    app = QApplication(sys.argv)
    w = InstallerWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
