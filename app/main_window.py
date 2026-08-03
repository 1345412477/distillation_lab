"""主窗口 - 左侧导航 + 右侧页面切换 (QStackedWidget) + 底部 GPU 状态"""
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QStackedWidget, QLabel, QFrame,
                              QStatusBar)
from PyQt6.QtCore import Qt, QSize
from .components.gpu_gauge import GpuGaugeWidget
from .components.gpu_detail_dialog import GpuDetailDialog
from .pages.dashboard import DashboardPage
from .pages.model_manager import ModelManagerPage
from .pages.distill_page import DistillPage
from .pages.prune_page import PrunePage
from .pages.monitor_page import MonitorPage
from .pages.compare_page import ComparePage
from .utils.theme import Theme


class SidebarButton(QPushButton):
    """侧边栏导航按钮"""

    def __init__(self, text, page_index=0):
        super().__init__(f"  {text}")
        self.page_index = page_index
        self.setCheckable(True)
        self.setMinimumHeight(44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.apply_theme()

    def apply_theme(self):
        t = Theme._colors[Theme.mode()]
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent; color: {t["text_secondary"]};
                border: none; border-radius: 6px;
                text-align: left; padding-left: 16px;
                font-size: 13px;
            }}
            QPushButton:hover {{ background-color: {t["bg_hover"]}; color: {t["text_primary"]}; }}
            QPushButton:checked {{
                background-color: {t["bg_surface"]}; color: {t["accent"]};
                font-weight: bold; border-left: 3px solid {t["accent"]};
            }}
        """)

    def sizeHint(self):
        return QSize(180, 44)


class MainWindow(QMainWindow):
    """应用主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DistillationLab - 模型蒸馏与剪枝实验平台")
        self.setMinimumSize(960, 640)
        self.resize(1280, 800)  # 默认启动大小，可自适应缩小
        self._current_index = 0
        self._theme_btn = None
        self._sidebar = None
        self._logo = None
        self._stack = None
        self._init_ui()
        self.apply_theme()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ==================== 左侧导航栏 ====================
        self._sidebar = QFrame()
        self._sidebar.setFixedWidth(200)
        sidebar_layout = QVBoxLayout(self._sidebar)
        sidebar_layout.setSpacing(4)
        sidebar_layout.setContentsMargins(8, 12, 8, 8)

        # Logo
        self._logo = QLabel("DistillationLab")
        self._logo.setStyleSheet("font-size: 14px; font-weight: bold; padding: 12px 8px 16px 8px;")
        sidebar_layout.addWidget(self._logo)

        # 导航按钮
        nav_items = [
            ("首页", 0),
            ("模型管理", 1),
            ("蒸馏实验", 2),
            ("剪枝实验", 3),
            ("实验监控", 4),
            ("实验对比", 5),
        ]

        self.nav_buttons = []
        for text, idx in nav_items:
            btn = SidebarButton(text, idx)
            btn.clicked.connect(lambda checked, i=idx: self._navigate(i))
            sidebar_layout.addWidget(btn)
            self.nav_buttons.append(btn)

        sidebar_layout.addStretch()

        # 主题切换按钮
        self._theme_btn = QPushButton("🌙 深色模式")
        self._theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._theme_btn.setMinimumHeight(36)
        self._theme_btn.clicked.connect(self._toggle_theme)
        sidebar_layout.addWidget(self._theme_btn)

        # 显卡详情（自动识别驱动/CUDA/温度/功耗/算力）
        self._gpu_detail_btn = QPushButton("🖥 显卡详情")
        self._gpu_detail_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._gpu_detail_btn.setMinimumHeight(36)
        self._gpu_detail_btn.clicked.connect(self._show_gpu_detail)
        sidebar_layout.addWidget(self._gpu_detail_btn)

        # GPU 状态仪表（每 2 秒刷新）
        self._gpu_gauge = GpuGaugeWidget()
        sidebar_layout.addWidget(self._gpu_gauge)

        sidebar_layout.addStretch()

        # 选中第一个按钮
        if self.nav_buttons:
            self.nav_buttons[0].setChecked(True)

        main_layout.addWidget(self._sidebar)

        # ==================== 右侧主内容区 ====================
        content_area = QVBoxLayout()
        content_area.setSpacing(0)
        content_area.setContentsMargins(0, 0, 0, 0)

        # 页面栈
        self._stack = QStackedWidget()

        self._pages = {}
        self._create_pages()

        content_area.addWidget(self._stack, 1)

        main_layout.addLayout(content_area, 1)

        # 状态栏
        status_bar = QStatusBar()
        status_bar.showMessage("就绪")
        self.setStatusBar(status_bar)

    def _create_pages(self):
        """创建所有页面"""
        # 首页
        dashboard = DashboardPage()
        dashboard.set_nav_callback(self._navigate)
        self._stack.addWidget(dashboard)
        self._pages["dashboard"] = dashboard

        # 模型管理
        model_mgr = ModelManagerPage()
        self._stack.addWidget(model_mgr)
        self._pages["models"] = model_mgr

        # 蒸馏实验
        distill = DistillPage()
        self._stack.addWidget(distill)
        self._pages["distill"] = distill

        # 剪枝实验
        prune = PrunePage()
        self._stack.addWidget(prune)
        self._pages["prune"] = prune

        # 实验监控
        monitor = MonitorPage()
        self._stack.addWidget(monitor)
        self._pages["monitor"] = monitor

        # 实验对比
        compare = ComparePage()
        self._stack.addWidget(compare)
        self._pages["compare"] = compare

    def _navigate(self, index):
        """切换到指定页面"""
        if 0 <= index < self._stack.count():
            self._stack.setCurrentIndex(index)
            self._current_index = index

            # 更新导航按钮状态
            for btn in self.nav_buttons:
                btn.setChecked(btn.page_index == index)

            # 页面刷新
            self._refresh_page(index)

    def _refresh_page(self, index):
        """刷新当前页面数据"""
        if index == 0:  # 首页
            self._pages["dashboard"].refresh()
        elif index == 1:  # 模型管理
            self._pages["models"].refresh()
        elif index == 2:  # 蒸馏
            self._pages["distill"].refresh_models()
        elif index == 3:  # 剪枝
            self._pages["prune"].refresh_models()
        elif index == 4:  # 监控
            self._pages["monitor"].refresh()
        elif index == 5:  # 对比
            self._pages["compare"].refresh()

    def apply_theme(self):
        """应用当前主题到主窗口"""
        t = Theme._colors[Theme.mode()]
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {t["bg_primary"]}; }}
            QWidget {{ color: {t["text_primary"]}; font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif; }}
        """)
        if self._sidebar:
            self._sidebar.setStyleSheet(f"QFrame {{ background-color: {t['bg_secondary']}; }}")
        if self._logo:
            self._logo.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {t['accent']}; padding: 12px 8px 16px 8px;")
        if self._stack:
            self._stack.setStyleSheet(f"background-color: {t['bg_primary']};")
        if self._theme_btn:
            mode = Theme.mode()
            self._theme_btn.setText("🌙 深色模式" if mode == "dark" else "️ 浅色模式")
            for btn in (self._theme_btn, self._gpu_detail_btn):
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: transparent; color: {t["text_secondary"]};
                        border: 1px solid {t["border"]}; border-radius: 6px;
                        text-align: center; font-size: 12px;
                    }}
                    QPushButton:hover {{
                        background-color: {t["bg_hover"]}; color: {t["text_primary"]};
                    }}
                """)
        # 更新导航按钮主题
        for btn in self.nav_buttons:
            btn.apply_theme()

    def _toggle_theme(self):
        """切换深色/浅色主题"""
        Theme.toggle()
        Theme.apply()
        self.apply_theme()
        # 更新所有子页面的主题
        for page in self._pages.values():
            if hasattr(page, "apply_theme"):
                try:
                    page.apply_theme()
                except:
                    pass

    def _show_gpu_detail(self):
        """打开显卡详情对话框"""
        dialog = GpuDetailDialog(self)
        dialog.exec()

    def closeEvent(self, event):
        """关闭窗口时清理资源"""
        for page in self._pages.values():
            if hasattr(page, "stop"):
                try:
                    page.stop()
                except:
                    pass
        if getattr(self, "_gpu_gauge", None):
            self._gpu_gauge.stop()
        super().closeEvent(event)
