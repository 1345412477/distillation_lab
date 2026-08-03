"""DistillationLab 应用入口"""
import sys
import os

# 确保项目根目录在导入路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from .main_window import MainWindow
from .utils.file_logger import setup_global_excepthook


def main():
    # 设置全局异常钩子（崩溃时写入文件日志）
    setup_global_excepthook()

    # 清理上次异常退出遗留的 running 实验
    from .models.db_manager import DBManager
    stale = DBManager().reset_stale_running()
    if stale:
        print(f"[DistillationLab] 已将 {stale} 个异常中断的实验标记为 failed")

    app = QApplication(sys.argv)
    app.setApplicationName("DistillationLab")
    app.setOrganizationName("DistillationLab")

    # 全局样式
    app.setStyle("Fusion")

    # GPU 后台轮询（避免 UI 线程频繁拉起 nvidia-smi 导致卡顿/弹窗）
    from .utils.gpu_monitor import start_gpu_poller
    start_gpu_poller(interval=2.0)

    from .utils.theme import Theme
    Theme.apply()

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
