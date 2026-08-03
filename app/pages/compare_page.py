"""实验对比页面 - 多实验选择、叠加对比、能力雷达图"""
import json
import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QCheckBox, QTableWidget,
                              QTableWidgetItem, QHeaderView, QGroupBox,
                              QSplitter)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImageReader, QPixmap, QIcon
from ..models.db_manager import DBManager
from ..components.loss_chart import LossChartWidget, LossChartWidgetWithContainer
from ..components.radar_chart import RadarChartWidget
from ..engines.evaluation import EvaluationEngine
from ..utils.theme import Theme


class ComparePage(QWidget):
    """实验对比页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.db = DBManager()
        self._checkboxes = {}
        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # 标题
        self.title = QLabel("实验对比")
        self.title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {Theme.get('text_primary')};")
        layout.addWidget(self.title)

        # 实验选择区域
        select_group = QGroupBox("选择对比实验")
        select_layout = QVBoxLayout(select_group)

        self.select_bar = QHBoxLayout()
        self.refresh_btn = QPushButton("刷新列表")
        self.refresh_btn.clicked.connect(self.refresh)
        self.refresh_btn.setStyleSheet(f"""
            QPushButton {{ background-color: {Theme.get("btn_default")}; color: {Theme.get("btn_default_text")};
                padding: 6px 14px; border-radius: 4px; }}
            QPushButton:hover {{ background-color: {Theme.get("btn_default_hover")}; }}
        """)
        self.select_bar.addStretch()
        self.select_bar.addWidget(self.refresh_btn)
        select_layout.addLayout(self.select_bar)

        self.checkbox_layout = QHBoxLayout()
        self.checkbox_layout.setSpacing(16)
        select_layout.addLayout(self.checkbox_layout)

        self.compare_btn = QPushButton("开始对比")
        self.compare_btn.clicked.connect(self._do_compare)
        self.compare_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.get("accent")}; color: {Theme.get("bg_primary")};
                padding: 8px 20px; border-radius: 4px; font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {Theme.get("accent_hover")}; }}
        """)
        select_layout.addWidget(self.compare_btn)
        layout.addWidget(select_group)

        # 对比结果区
        splitter = QSplitter(Qt.Orientation.Vertical)

        # 上方：Loss 曲线 + 雷达图
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)

        self.loss_chart = LossChartWidgetWithContainer("训练 Loss 对比")
        top_layout.addWidget(self.loss_chart, 2)

        self.radar_chart = RadarChartWidget("能力对比")
        self.radar_chart.setMinimumWidth(300)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.radar_chart, 3)

        scores_group = QGroupBox("维度评分（0-100）")
        scores_layout = QVBoxLayout(scores_group)
        self.scores_table = QTableWidget()
        self.scores_table.setColumnCount(0)
        self.scores_table.setStyleSheet(f"""
            QTableWidget {{ background-color: {Theme.get("bg_secondary")}; color: {Theme.get("text_primary")};
                border: 1px solid {Theme.get("border")}; gridline-color: {Theme.get("border")}; }}
            QHeaderView::section {{ background-color: {Theme.get("header_bg")}; color: {Theme.get("text_primary")};
                padding: 4px; border: 1px solid {Theme.get("border")}; font-weight: bold; }}
        """)
        scores_layout.addWidget(self.scores_table)
        right_layout.addWidget(scores_group, 2)
        top_layout.addWidget(right_panel, 1)

        splitter.addWidget(top_widget)

        # 下方：对比表格
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)

        table_group = QGroupBox("模型回答对比")
        table_layout = QVBoxLayout(table_group)

        self.compare_table = QTableWidget()
        self.compare_table.setColumnCount(3)
        self.compare_table.setHorizontalHeaderLabels(["问题", "基线", "实验组"])
        self.compare_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.compare_table.setStyleSheet(f"""
            QTableWidget {{ background-color: {Theme.get("bg_secondary")}; color: {Theme.get("text_primary")};
                border: 1px solid {Theme.get("border")}; gridline-color: {Theme.get("border")}; }}
            QHeaderView::section {{ background-color: {Theme.get("header_bg")}; color: {Theme.get("text_secondary")};
                padding: 4px; border: 1px solid {Theme.get("border")}; font-weight: bold; }}
        """)
        table_layout.addWidget(self.compare_table)
        bottom_layout.addWidget(table_group)

        splitter.addWidget(bottom_widget)
        layout.addWidget(splitter)

    def refresh(self):
        """刷新实验列表"""
        # 清除旧的 checkbox
        for cb in list(self._checkboxes.values()):
            self.checkbox_layout.removeWidget(cb)
            cb.deleteLater()
        self._checkboxes.clear()

        experiments = self.db.get_all_experiments()
        for exp in experiments:
            type_label = {"distillation": "蒸馏", "pruning": "剪枝"}.get(exp["type"], exp["type"])
            status_icons = {
                "pending": "⏳", "running": "🔄", "completed": "✅", "failed": "❌"
            }
            icon = status_icons.get(exp["status"], "❓")
            cb = QCheckBox(f"{icon} #{exp['id']} {exp.get('name','')} ({type_label})")
            cb.setStyleSheet(f"color: {Theme.get('text_primary')};")
            cb.exp_id = exp["id"]
            self._checkboxes[exp["id"]] = cb
            self.checkbox_layout.addWidget(cb)

    def _do_compare(self):
        """执行对比分析"""
        selected_ids = [
            cb.exp_id for cb in self._checkboxes.values() if cb.isChecked()
        ]

        if not selected_ids:
            self.compare_table.setRowCount(0)
            self.loss_chart.chart.clear_series()
            self.radar_chart.clear()
            return

        # 收集选中的实验数据
        all_loss_data = {}
        radar_series = {}
        radar_dims = []
        exp_baselines = {}

        for exp_id in selected_ids:
            exp = self.db.get_experiment_by_id(exp_id)
            if not exp:
                continue

            label = f"#{exp_id} ({'蒸馏' if exp['type']=='distillation' else '剪枝'})"
            name = f"#{exp_id}"

            try:
                metrics = json.loads(exp["metrics"]) if isinstance(exp["metrics"], str) else exp["metrics"]
                eval_results = json.loads(exp["eval_results"]) if isinstance(exp["eval_results"], str) else exp["eval_results"]
            except (json.JSONDecodeError, TypeError):
                metrics = []
                eval_results = {}

            # 收集 Loss 数据（按 step 绘制）
            if metrics:
                loss_values = []
                for m in metrics:
                    if isinstance(m, dict):
                        loss_values.append(m.get("avg_loss", m.get("loss", 0)))
                if loss_values:
                    all_loss_data[name] = loss_values

            baseline = eval_results.get("baseline", [])
            after = eval_results.get("after", eval_results.get("recovered", []))
            teacher = eval_results.get("teacher", [])
            exp_baselines[name] = baseline

            # 雷达图：教师/蒸馏前/蒸馏后 三系列真实维度评分
            try:
                cfg = json.loads(exp["config"]) if isinstance(exp["config"], str) else exp["config"]
                references = cfg.get("reference_answers") or []
            except (json.JSONDecodeError, TypeError):
                references = []
            scores = eval_results.get("scores")
            if not scores and baseline and after:
                scores = EvaluationEngine.compute_scores(baseline, after, references)
            if scores and scores.get("dimensions"):
                radar_dims = scores["dimensions"]
                radar_series[f"{name} 蒸馏前"] = scores["baseline"]
                radar_series[f"{name} 蒸馏后"] = scores["after"]
                if teacher:
                    teacher_series = scores.get("teacher")
                    if not teacher_series and scores.get("dimensions"):
                        t_scores = EvaluationEngine.compute_scores(teacher, teacher, references)
                        teacher_series = t_scores["baseline"]
                    radar_series[f"{name} 教师"] = teacher_series or []

        # 更新 Loss 曲线
        self.loss_chart.chart.clear_series()
        for name, values in all_loss_data.items():
            for i, v in enumerate(values):
                self.loss_chart.chart.update_data(name, v, i)

        # 更新雷达图（真实能力维度）
        if radar_series:
            self.radar_chart.set_data(radar_dims, radar_series)
            self._fill_scores_table(radar_dims, radar_series)
        else:
            self.radar_chart.clear()
            self.scores_table.setRowCount(0)
            self.scores_table.setColumnCount(0)

        self._fill_compare_table(selected_ids, exp_baselines)

    def _fill_scores_table(self, dims, series):
        """填充维度评分表：行=模型系列，列=评分维度"""
        self.scores_table.setColumnCount(len(dims) + 1)
        self.scores_table.setHorizontalHeaderLabels(["模型"] + dims)
        self.scores_table.setRowCount(len(series))
        for row, (name, values) in enumerate(series.items()):
            self.scores_table.setItem(row, 0, QTableWidgetItem(name))
            for col, v in enumerate(values):
                self.scores_table.setItem(row, col + 1, QTableWidgetItem(f"{v:.1f}"))
        self.scores_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def _fill_compare_table(self, selected_ids, exp_baselines):
        """填充逐题回答对比表（教师/蒸馏前/蒸馏后）"""
        # 更新对比表格：第一个实验的 baseline 作为参考
        first_exp = None
        for exp_id in selected_ids:
            exp = self.db.get_experiment_by_id(exp_id)
            if exp:
                first_exp = exp
                break

        if first_exp:
            try:
                eval_results = json.loads(first_exp["eval_results"]) if isinstance(first_exp["eval_results"], str) else first_exp["eval_results"]
            except (json.JSONDecodeError, TypeError):
                eval_results = {}
            baseline = eval_results.get("baseline", [])
            if not baseline:
                baseline = exp_baselines.get(f"#{first_exp['id']}", [])
            teacher = eval_results.get("teacher", [])
            has_images = any(b.get("image") for b in baseline if b)

            all_after = []
            for exp_id in selected_ids:
                exp = self.db.get_experiment_by_id(exp_id)
                if exp:
                    try:
                        er = json.loads(exp["eval_results"]) if isinstance(exp["eval_results"], str) else exp["eval_results"]
                    except (json.JSONDecodeError, TypeError):
                        er = {}
                    after = er.get("after", er.get("recovered", []))
                    all_after.append((f"#{exp_id}", after))

            if teacher:
                self.compare_table.setColumnCount(3 + len(all_after) + (1 if has_images else 0))
                headers = (["图片"] if has_images else []) + ["问题", "教师", "蒸馏前"] + [f"{name} 蒸馏后" for name, _ in all_after]
            else:
                self.compare_table.setColumnCount(2 + len(all_after) + (1 if has_images else 0))
                headers = (["图片"] if has_images else []) + ["问题", "基线"] + [name for name, _ in all_after]
            self.compare_table.setHorizontalHeaderLabels(headers)
            self.compare_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

            self.compare_table.setRowCount(len(baseline))
            for row, b in enumerate(baseline):
                col = 0
                if has_images:
                    img_item = QTableWidgetItem(b.get("image", ""))
                    img_path = b.get("image_path", "")
                    if img_path and os.path.isfile(img_path):
                        reader = QImageReader(img_path)
                        reader.setScaledSize(QSize(40, 40))
                        pixmap = QPixmap.fromImage(reader.read())
                        if not pixmap.isNull():
                            img_item.setIcon(QIcon(pixmap))
                    self.compare_table.setItem(row, col, img_item)
                    col += 1
                self.compare_table.setItem(row, col, QTableWidgetItem(b.get("question", "")[:40]))
                col += 1
                if teacher:
                    t = teacher[row] if row < len(teacher) and teacher[row] else {}
                    self.compare_table.setItem(row, col, QTableWidgetItem(t.get("answer", "")[:60]))
                    col += 1
                self.compare_table.setItem(row, col, QTableWidgetItem(b.get("answer", "")[:60]))
                col += 1
                for _, after_results in all_after:
                    if row < len(after_results) and after_results[row]:
                        self.compare_table.setItem(
                            row, col,
                            QTableWidgetItem(after_results[row].get("answer", "")[:60]),
                        )
                    col += 1

    def apply_theme(self):
        """应用当前主题"""
        t = Theme._colors[Theme.mode()]
        self.setStyleSheet(f"background-color: {t['bg_primary']};")
        
        # 更新标题 - 确保高对比度
        self.title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {t['text_primary']}; padding: 8px 0;")
        
        # 更新 GroupBox 样式 - 确保有背景色
        for group_box in self.findChildren(QGroupBox):
            group_box.setStyleSheet(f"""
                QGroupBox {{
                    font-weight: bold;
                    color: {t['text_primary']};
                    border: 1px solid {t['border']};
                    border-radius: 6px;
                    margin-top: 8px;
                    padding: 12px 8px 8px 8px;
                    background-color: {t['bg_tertiary']};
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    padding: 2px 8px;
                    color: {t['text_primary']};
                }}
            """)
        
        # 更新按钮 - 浅色模式下使用更明显的颜色
        self.refresh_btn.setStyleSheet(f"""
            QPushButton {{ 
                background-color: {t["btn_default"]}; 
                color: {t["btn_default_text"]};
                padding: 6px 14px; 
                border-radius: 4px;
                border: 1px solid {t["border"]};
            }}
            QPushButton:hover {{ 
                background-color: {t["btn_default_hover"]}; 
            }}
        """)
        
        # 开始对比按钮 - 白色文字确保对比度
        self.compare_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {t["accent"]};
                color: #ffffff;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {t["accent_hover"]};
            }}
        """)
        
        # 更新 checkbox
        for cb in self._checkboxes.values():
            cb.setStyleSheet(f"color: {t['text_primary']};")
        
        # 更新表格 - 浅色模式下使用更清晰的配色
        self.compare_table.setStyleSheet(f"""
            QTableWidget {{ 
                background-color: {t["bg_card"]}; 
                color: {t["text_primary"]};
                border: 1px solid {t["border"]}; 
                gridline-color: {t["border"]};
                alternate-background-color: {t["table_alt"]};
            }}
            QTableWidget::item {{ 
                padding: 6px; 
            }}
            QHeaderView::section {{ 
                background-color: {t["header_bg"]}; 
                color: {t["text_primary"]};
                padding: 8px; 
                border: 1px solid {t["border"]}; 
                font-weight: bold;
            }}
        """)
        
        # 更新图表主题
        self.loss_chart.apply_theme()
        self.radar_chart.apply_theme()
