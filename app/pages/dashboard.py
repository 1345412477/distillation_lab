"""首页概览 - 实验统计、快速入口、GPU 状态"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QFrame, QSizePolicy, QPushButton, QMessageBox)
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt
from ..models.db_manager import DBManager
from ..utils.theme import Theme
from ..utils.gpu_monitor import get_gpu_info, get_cuda_runtime_info


class StatCard(QFrame):
    """统计卡片 - 使用 QFrame + 纯 stylesheet 确保背景色可靠渲染"""

    def __init__(self, title_text, value, color, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(180)
        self._title_text = title_text
        self._value = value
        self._accent_color = color
        self._build()

    def _build(self):
        bg = Theme.get("bg_card")
        border = Theme.get("border")
        text_sec = Theme.get("text_secondary")

        # 关键：background-color 必须在 stylesheet 中显式指定
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg};
                border: 1px solid {border};
                border-left: 4px solid {self._accent_color};
                border-radius: 8px;
            }}
        """)

        # 清除旧子控件
        if self.layout():
            while self.layout().count():
                item = self.layout().takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(14, 12, 14, 12)

        icon_label = QLabel(self._title_text)
        icon_label.setStyleSheet(f"font-size: 13px; color: {text_sec}; font-weight: 500;")
        layout.addWidget(icon_label)

        value_label = QLabel(self._value)
        value_label.setStyleSheet(f"font-size: 32px; font-weight: bold; color: {self._accent_color};")
        layout.addWidget(value_label)


class DashboardPage(QWidget):
    """首页概览页面"""

    NAVIGATION_REQUESTED = "__nav__"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.db = DBManager()
        self._nav_callback = None
        self.init_ui()

    def set_nav_callback(self, callback):
        self._nav_callback = callback

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # 标题
        self.title = QLabel("DistillationLab")
        self.title.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {Theme.get('text_primary')};")
        layout.addWidget(self.title)

        self.subtitle = QLabel("模型蒸馏与剪枝实验平台")
        self.subtitle.setStyleSheet(f"font-size: 14px; color: {Theme.get('text_secondary')}; margin-bottom: 12px;")
        layout.addWidget(self.subtitle)

        # 统计卡片区域
        self.stats_layout = QHBoxLayout()
        self.stats_layout.setSpacing(12)
        layout.addLayout(self.stats_layout)
        self._build_stat_cards()

        # 硬件信息
        self.hw_section = QLabel("本机硬件")
        self.hw_section.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {Theme.get('text_primary')}; padding-top: 8px;")
        layout.addWidget(self.hw_section)
        self.hw_label = QLabel("-")
        self.hw_label.setWordWrap(True)
        self.hw_label.setStyleSheet(f"color: {Theme.get('text_secondary')}; font-size: 12px;")
        layout.addWidget(self.hw_label)

        # 实验列表
        self.recent_section = QLabel("最近实验")
        self.recent_section.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {Theme.get('text_primary')}; padding-top: 8px;")
        layout.addWidget(self.recent_section)

        self._recent_list = QVBoxLayout()
        layout.addLayout(self._recent_list)

        layout.addStretch()

    def _get_stats(self):
        """获取统计数据"""
        models = self.db.get_all_models()
        experiments = self.db.get_all_experiments()

        return {
            "models": len(models),
            "distill_count": sum(1 for e in experiments if e["type"] == "distillation"),
            "prune_count": sum(1 for e in experiments if e["type"] == "pruning"),
            "running": sum(1 for e in experiments if e["status"] == "running"),
            "recent": experiments[:5] if experiments else [],
        }

    def _build_stat_cards(self):
        """构建统计卡片（主题切换时重建）"""
        stats = self._get_stats()
        # 根据主题使用不同的强调色 - 浅色模式用深色确保对比度
        if Theme.mode() == "light":
            cards = [
                ("已注册模型", str(stats["models"]), "#2563eb"),   # 蓝色
                ("蒸馏实验", str(stats["distill_count"]), "#16a34a"),  # 绿色
                ("剪枝实验", str(stats["prune_count"]), "#ca8a04"),  # 黄色
                ("进行中", str(stats["running"]), "#dc2626"),     # 红色
            ]
        else:
            cards = [
                ("已注册模型", str(stats["models"]), "#89b4fa"),
                ("蒸馏实验", str(stats["distill_count"]), "#a6e3a1"),
                ("剪枝实验", str(stats["prune_count"]), "#f9e2af"),
                ("进行中", str(stats["running"]), "#f38ba8"),
            ]
        for title_text, value, color in cards:
            card = StatCard(title_text, value, color)
            self.stats_layout.addWidget(card)

    def refresh(self):
        """刷新页面"""
        self._refresh_hardware()
        stats = self._get_stats()
        # 清空最近实验列表
        for i in reversed(range(self._recent_list.count())):
            widget = self._recent_list.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        for exp in stats["recent"]:
            # 根据主题使用不同的状态颜色
            if Theme.mode() == "light":
                status_colors = {
                    "pending": "#7c8294",
                    "running": "#ca8a04",
                    "completed": "#16a34a",
                    "failed": "#dc2626",
                }
            else:
                status_colors = {
                    "pending": "#6c7086",
                    "running": "#f9e2af",
                    "completed": "#a6e3a1",
                    "failed": "#f38ba8",
                }
            color = status_colors.get(exp["status"], "#6c7086")
            exp_name = exp.get("name", f"实验 #{exp['id']}")
            exp_type_label = {"distillation": "蒸馏", "pruning": "剪枝"}.get(exp["type"], exp["type"])

            item = QFrame()
            item.setStyleSheet(f"QFrame {{ background-color: {Theme.get('bg_tertiary')}; border-radius: 4px; }}")
            item_layout = QHBoxLayout(item)
            item_layout.setContentsMargins(12, 6, 12, 6)

            status_dot = QLabel("●")
            status_dot.setStyleSheet(f"color: {color}; font-size: 14px;")
            item_layout.addWidget(status_dot)

            name_label = QLabel(f"{exp_name} ({exp_type_label})")
            name_label.setStyleSheet(f"color: {Theme.get('text_primary')}; font-size: 12px;")
            item_layout.addWidget(name_label)

            item_layout.addStretch()

            status_text = {
                "pending": "待执行",
                "running": "进行中",
                "completed": "已完成",
                "failed": "失败",
            }.get(exp["status"], exp["status"])
            status_label = QLabel(status_text)
            status_label.setStyleSheet(f"color: {color}; font-size: 11px;")
            item_layout.addWidget(status_label)

            # 删除按钮
            delete_btn = QPushButton("删除")
            delete_btn.setMaximumWidth(44)
            delete_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {Theme.get("text_muted")};
                    border: 1px solid {Theme.get("border")};
                    border-radius: 3px;
                    font-size: 10px;
                    padding: 1px 6px;
                }}
                QPushButton:hover {{
                    color: {Theme.get("red")};
                    border-color: {Theme.get("red")};
                }}
            """)
            delete_btn.clicked.connect(
                lambda checked, eid=exp["id"]: self._delete_experiment(eid)
            )
            item_layout.addWidget(delete_btn)

            self._recent_list.addWidget(item)

    def _delete_experiment(self, exp_id):
        """删除指定实验（含其蒸馏数据），确认后刷新"""
        exp = self.db.get_experiment_by_id(exp_id)
        if not exp:
            return
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除实验 #{exp_id}（{exp.get('name', '')}）吗？\n"
            "此操作不可恢复，将同时删除相关蒸馏数据。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.db.delete_experiment(exp_id)
        self.refresh()

    def _refresh_hardware(self):
        """刷新硬件摘要（GPU 名称/显存/温度/CUDA）"""
        gpus = get_gpu_info()
        if not gpus:
            self.hw_label.setText("未检测到 NVIDIA GPU（当前使用 CPU 模式）")
            return
        runtime = get_cuda_runtime_info()
        parts = []
        for g in gpus:
            mem_pct = (g["memory_used"] / g["memory_total"] * 100) if g["memory_total"] > 0 else 0
            text = (
                f"GPU {g['index']}: {g['name']} | "
                f"显存 {g['memory_used']:.1f}/{g['memory_total']:.1f} GB ({mem_pct:.0f}%) | "
                f"利用率 {g.get('utilization', 0)}%"
            )
            if g.get("temperature"):
                text += f" | {g['temperature']}°C"
            parts.append(text)
        parts.append(f"PyTorch {runtime['torch']} / CUDA {runtime['cuda']}")
        self.hw_label.setText("\n".join(parts))

    def apply_theme(self):
        """应用当前主题"""
        self.setStyleSheet("")
        self.title.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {Theme.get('text_primary')};")
        self.subtitle.setStyleSheet(f"font-size: 14px; color: {Theme.get('text_secondary')}; margin-bottom: 12px;")
        self.recent_section.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {Theme.get('text_primary')}; padding-top: 8px;")
        self.hw_section.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {Theme.get('text_primary')}; padding-top: 8px;")
        self.hw_label.setStyleSheet(f"color: {Theme.get('text_secondary')}; font-size: 12px;")
        # 重建统计卡片（使用新主题颜色）
        self._rebuild_stat_cards()
        # 刷新最近实验列表
        self.refresh()

    def _rebuild_stat_cards(self):
        """主题切换时重建统计卡片"""
        # 清除旧卡片
        while self.stats_layout.count():
            item = self.stats_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # 重建
        self._build_stat_cards()
