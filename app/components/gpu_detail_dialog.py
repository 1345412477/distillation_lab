"""显卡详情对话框 - 自动识别并实时刷新本机 GPU 详细资料"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QProgressBar,
                              QFrame, QHBoxLayout, QPushButton)
from PyQt6.QtCore import Qt, QTimer
from ..utils.gpu_monitor import get_gpu_info, get_cuda_runtime_info
from ..utils.theme import Theme


class GpuDetailDialog(QDialog):
    """展示每张 GPU 的详细参数（驱动/CUDA/显存/温度/功耗/算力）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("显卡详情")
        self.setMinimumWidth(460)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(3000)
        self._init_ui()
        self.refresh()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self.runtime_label = QLabel("-")
        self.runtime_label.setStyleSheet(
            f"color: {Theme.get('text_secondary')}; font-size: 12px;"
        )
        layout.addWidget(self.runtime_label)

        self.card_layout = QVBoxLayout()
        self.card_layout.setSpacing(8)
        layout.addLayout(self.card_layout)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.get("btn_default")};
                color: {Theme.get("btn_default_text")};
                padding: 6px 18px; border-radius: 4px;
            }}
            QPushButton:hover {{ background-color: {Theme.get("btn_default_hover")}; }}
        """)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _clear_cards(self):
        while self.card_layout.count():
            item = self.card_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def refresh(self):
        gpus = get_gpu_info()
        runtime = get_cuda_runtime_info()
        self.runtime_label.setText(
            f"PyTorch {runtime['torch']} | CUDA {runtime['cuda']} | cuDNN {runtime['cudnn']}"
        )
        self._clear_cards()

        if not gpus:
            label = QLabel("未检测到 NVIDIA GPU（当前使用 CPU 模式）")
            label.setStyleSheet(f"color: {Theme.get('text_muted')}; font-size: 12px;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.card_layout.addWidget(label)
            return

        for gpu in gpus:
            self.card_layout.addWidget(self._build_card(gpu))

    def _build_card(self, gpu):
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {Theme.get("bg_card")};
                border: 1px solid {Theme.get("border")};
                border-radius: 8px;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setSpacing(4)
        layout.setContentsMargins(12, 10, 12, 10)

        title = QLabel(f"GPU {gpu['index']}: {gpu['name']}")
        title.setStyleSheet(f"color: {Theme.get('text_primary')}; font-size: 13px; font-weight: bold;")
        layout.addWidget(title)

        meta = (
            f"驱动 {gpu.get('driver_version', 'unknown')} | "
            f"算力 {gpu.get('compute_cap', 'unknown')}"
        )
        if gpu.get("temperature"):
            meta += f" | {gpu['temperature']}°C"
        if gpu.get("power_w"):
            meta += f" | {gpu['power_w']:.0f} W"
        meta_label = QLabel(meta)
        meta_label.setStyleSheet(f"color: {Theme.get('text_muted')}; font-size: 11px;")
        layout.addWidget(meta_label)

        mem_pct = (gpu["memory_used"] / gpu["memory_total"] * 100) if gpu["memory_total"] > 0 else 0
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(int(mem_pct))
        bar.setFormat(
            f"显存 {gpu['memory_used']:.1f} / {gpu['memory_total']:.1f} GB "
            f"（空闲 {gpu.get('memory_free', 0):.1f} GB，{mem_pct:.0f}%）"
        )
        bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {Theme.get("bg_surface")}; border: none;
                border-radius: 3px; height: 18px; text-align: center;
                color: {Theme.get("text_primary")}; font-size: 10px;
            }}
            QProgressBar::chunk {{
                background-color: {Theme.get("red") if mem_pct > 90 else Theme.get("yellow") if mem_pct > 70 else Theme.get("accent")};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(bar)

        util = QProgressBar()
        util.setRange(0, 100)
        util.setValue(int(gpu.get("utilization", 0)))
        util.setFormat(f"利用率 {gpu.get('utilization', 0)}%")
        util.setStyleSheet(f"""
            QProgressBar {{
                background-color: {Theme.get("bg_surface")}; border: none;
                border-radius: 3px; height: 18px; text-align: center;
                color: {Theme.get("text_primary")}; font-size: 10px;
            }}
            QProgressBar::chunk {{
                background-color: {Theme.get("green")}; border-radius: 3px;
            }}
        """)
        layout.addWidget(util)
        return card

    def stop(self):
        self._timer.stop()

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)
