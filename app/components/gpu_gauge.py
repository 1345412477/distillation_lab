"""GPU 仪表盘 - 显示 GPU 显存和利用率"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout,
                              QLabel, QProgressBar, QFrame)
from PyQt6.QtCore import Qt, QTimer
from ..utils.gpu_monitor import get_gpu_info
from ..utils.theme import Theme


class GpuGaugeWidget(QWidget):
    """GPU 状态显示组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gpus = []

        # 定时刷新
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(3000)  # 每3秒刷新（读后台缓存，不阻塞 UI）

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        self._header = QLabel("GPU 状态")
        layout.addWidget(self._header)

        self._card_layout = QVBoxLayout()
        self._card_layout.setSpacing(6)
        layout.addLayout(self._card_layout)

        self.apply_theme()
        self.refresh()

    def apply_theme(self):
        """应用当前主题"""
        self.setStyleSheet(f"""
            GpuGaugeWidget {{
                background-color: {Theme.get("bg_secondary")};
                border: 1px solid {Theme.get("border")};
                border-radius: 6px;
            }}
        """)
        self._header.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {Theme.get('text_primary')};")

    def refresh(self):
        """刷新 GPU 状态"""
        gpus = get_gpu_info()
        if gpus is None:
            self._show_no_gpu()
            return

        self._gpus = gpus
        self._render_gpu_cards()

    def _show_no_gpu(self):
        """显示无 GPU"""
        for i in reversed(range(self._card_layout.count())):
            widget = self._card_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        label = QLabel("未检测到 GPU（使用 CPU 模式）")
        label.setStyleSheet(f"color: {Theme.get('text_muted')}; font-size: 11px; padding: 8px;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._card_layout.addWidget(label)

    def _render_gpu_cards(self):
        """渲染 GPU 卡片"""
        for i in reversed(range(self._card_layout.count())):
            widget = self._card_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        for gpu in self._gpus:
            card = self._create_gpu_card(gpu)
            self._card_layout.addWidget(card)

    def _create_gpu_card(self, gpu):
        """创建单个 GPU 状态卡片"""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {Theme.get("bg_card")};
                border: 1px solid {Theme.get("border")};
                border-radius: 4px;
            }}
        """)

        layout = QVBoxLayout(card)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 6, 8, 6)

        # GPU 名称
        name_label = QLabel(f"GPU {gpu['index']}: {gpu['name'][:30]}")
        name_label.setStyleSheet(f"color: {Theme.get('text_primary')}; font-size: 11px; font-weight: bold;")
        layout.addWidget(name_label)

        # 显存使用
        mem_pct = (gpu["memory_used"] / gpu["memory_total"] * 100) if gpu["memory_total"] > 0 else 0
        mem_bar = QProgressBar()
        mem_bar.setRange(0, 100)
        mem_bar.setValue(int(mem_pct))
        mem_bar.setTextVisible(True)
        mem_bar.setFormat(
            f"显存: {gpu['memory_used']:.1f}/{gpu['memory_total']:.1f} GB ({mem_pct:.0f}%)"
        )
        mem_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {Theme.get("bg_surface")}; border: none; border-radius: 3px;
                height: 18px; text-align: center; color: {Theme.get("text_primary")}; font-size: 10px;
            }}
            QProgressBar::chunk {{
                background-color: {self._mem_color(mem_pct)};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(mem_bar)

        # 利用率
        util = gpu.get("utilization", 0)
        util_bar = QProgressBar()
        util_bar.setRange(0, 100)
        util_bar.setValue(util)
        util_bar.setTextVisible(True)
        util_bar.setFormat(f"利用率: {util}%")
        util_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {Theme.get("bg_surface")}; border: none; border-radius: 3px;
                height: 18px; text-align: center; color: {Theme.get("text_primary")}; font-size: 10px;
            }}
            QProgressBar::chunk {{
                background-color: {self._util_color(util)};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(util_bar)

        # 温度 / 功耗摘要
        extras = []
        if gpu.get("temperature"):
            extras.append(f"{gpu['temperature']}°C")
        if gpu.get("power_w"):
            extras.append(f"{gpu['power_w']:.0f}W")
        if extras:
            extra_label = QLabel(" | ".join(extras))
            extra_label.setStyleSheet(
                f"color: {Theme.get('text_muted')}; font-size: 10px;"
            )
            layout.addWidget(extra_label)

        return card

    def _mem_color(self, pct):
        if pct > 90:
            return Theme.get("red")
        elif pct > 70:
            return Theme.get("yellow")
        return Theme.get("accent")

    def _util_color(self, util):
        if util > 90:
            return Theme.get("green")
        elif util > 50:
            return Theme.get("yellow")
        return Theme.get("text_muted")

    def stop(self):
        """停止刷新"""
        self._timer.stop()
