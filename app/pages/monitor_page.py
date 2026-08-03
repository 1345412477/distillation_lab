"""实时监控页面 - 实验详情、Loss 曲线、日志"""
import os
import json
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QComboBox, QTextEdit, QGroupBox,
                              QFormLayout, QTableWidget,
                              QTableWidgetItem, QHeaderView, QSplitter,
                              QMessageBox, QListWidget, QListWidgetItem)
from PyQt6.QtCore import Qt, QSize, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtGui import QImageReader, QPixmap, QIcon
from ..models.db_manager import DBManager
from ..components.loss_chart import LossChartWidget, LossChartWidgetWithContainer
from ..utils.theme import Theme


class MonitorPage(QWidget):
    """实验详情与实时监控页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.db = DBManager()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # 标题和实验选择
        header_layout = QHBoxLayout()
        self.title = QLabel("实验监控")
        self.title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {Theme.get('text_primary')};")
        header_layout.addWidget(self.title)

        header_layout.addStretch()

        self.exp_combo = QComboBox()
        self.exp_combo.setMinimumWidth(340)
        self.exp_combo.setMinimumHeight(32)
        self.exp_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {Theme.get("bg_secondary")};
                color: {Theme.get("text_primary")};
                border: 1px solid {Theme.get("border")};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }}
            QComboBox:hover {{
                border-color: {Theme.get("accent")};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
            QComboBox::down-arrow {{
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {Theme.get("text_secondary")};
                margin-right: 6px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {Theme.get("bg_secondary")};
                color: {Theme.get("text_primary")};
                border: 1px solid {Theme.get("border")};
                selection-background-color: {Theme.get("accent")};
                selection-color: {Theme.get("bg_primary")};
                padding: 4px;
            }}
        """)
        self.exp_combo.currentIndexChanged.connect(self._on_exp_changed)
        header_layout.addWidget(QLabel("选择实验:"))
        header_layout.addWidget(self.exp_combo)

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh)
        self.refresh_btn.setStyleSheet(f"""
            QPushButton {{ background-color: {Theme.get("btn_default")}; color: {Theme.get("btn_default_text")};
                padding: 6px 14px; border-radius: 4px; }}
            QPushButton:hover {{ background-color: {Theme.get("btn_default_hover")}; }}
        """)
        header_layout.addWidget(self.refresh_btn)

        self.delete_btn = QPushButton("删除实验")
        self.delete_btn.setVisible(False)
        self.delete_btn.clicked.connect(self._delete_current)
        self.delete_btn.setStyleSheet(f"""
            QPushButton {{ background-color: {Theme.get("red")}; color: {Theme.get("bg_primary")};
                padding: 6px 14px; border-radius: 4px; font-weight: bold; }}
            QPushButton:hover {{ background-color: {Theme.get("red")}; }}
        """)
        header_layout.addWidget(self.delete_btn)

        self.open_report_btn = QPushButton("打开评估报告")
        self.open_report_btn.setEnabled(False)
        self.open_report_btn.clicked.connect(self._open_report)
        self.open_report_btn.setStyleSheet(f"""
            QPushButton {{ background-color: {Theme.get("accent")}; color: #ffffff;
                padding: 6px 14px; border-radius: 4px; font-weight: bold; }}
            QPushButton:hover {{ background-color: {Theme.get("accent_hover")}; }}
        """)
        header_layout.addWidget(self.open_report_btn)

        layout.addLayout(header_layout)

        # 主内容区
        splitter = QSplitter(Qt.Orientation.Vertical)

        # 上半部分：Loss 曲线 + 实验信息
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)

        # Loss 曲线
        self.loss_chart_container = LossChartWidgetWithContainer("Loss 曲线")
        top_layout.addWidget(self.loss_chart_container, 2)

        # 实验信息面板
        info_panel = QGroupBox("实验信息")
        info_layout = QFormLayout(info_panel)
        self.exp_info_labels = {}
        fields = ["状态", "实验类型", "蒸馏模式", "教师/目标模型", "学生模型",
                  "图像目录", "蒸馏数据", "维度评分", "输出目录", "开始时间", "完成时间"]
        for f in fields:
            label = QLabel("-")
            label.setStyleSheet(f"color: {Theme.get('text_primary')};")
            label.setWordWrap(True)
            info_layout.addRow(f"{f}:", label)
            self.exp_info_labels[f] = label

        top_layout.addWidget(info_panel, 1)
        splitter.addWidget(top_widget)

        # 下半部分：评估结果 + 日志
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)

        # 评估对比表格
        eval_group = QGroupBox("评估结果对比")
        eval_group_layout = QVBoxLayout(eval_group)
        self.eval_table = QTableWidget()
        self.eval_table.setColumnCount(4)
        self.eval_table.setHorizontalHeaderLabels(["问题", "基线", "处理后", "对比分析"])
        self.eval_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.eval_table.setStyleSheet(f"""
            QTableWidget {{ background-color: {Theme.get("bg_secondary")}; color: {Theme.get("text_primary")};
                border: 1px solid {Theme.get("border")}; gridline-color: {Theme.get("border")}; }}
            QHeaderView::section {{ background-color: {Theme.get("header_bg")}; color: {Theme.get("text_primary")};
                padding: 4px; border: 1px solid {Theme.get("border")}; font-weight: bold; }}
        """)
        eval_group_layout.addWidget(self.eval_table)
        bottom_layout.addWidget(eval_group, 2)

        # 日志
        log_group = QGroupBox("实时日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{ background-color: {Theme.get("bg_log")}; color: {Theme.get("text_secondary")};
                border: none; font-family: 'Consolas', monospace; font-size: 11px; }}
        """)
        log_layout.addWidget(self.log_text)
        bottom_layout.addWidget(log_group, 1)

        splitter.addWidget(bottom_widget)

        # 逐图识别详情（多模态实验）
        detail_widget = QWidget()
        detail_layout = QHBoxLayout(detail_widget)
        img_group = QGroupBox("图片列表")
        img_layout = QVBoxLayout(img_group)
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._on_image_selected)
        img_layout.addWidget(self.image_list)
        detail_layout.addWidget(img_group, 1)
        ans_group = QGroupBox("识别详情（教师 / 蒸馏前 / 蒸馏后）")
        ans_layout = QVBoxLayout(ans_group)
        self.image_detail_text = QTextEdit()
        self.image_detail_text.setReadOnly(True)
        ans_layout.addWidget(self.image_detail_text)
        detail_layout.addWidget(ans_group, 2)
        splitter.addWidget(detail_widget)
        self._image_groups = []
        layout.addWidget(splitter)

    def refresh(self):
        """刷新实验列表"""
        current_id = self.exp_combo.currentData()
        self.exp_combo.blockSignals(True)
        try:
            self.exp_combo.clear()

            experiments = self.db.get_all_experiments()
            for exp in experiments:
                type_label = {"distillation": "蒸馏", "pruning": "剪枝"}.get(exp["type"], exp["type"])
                label = f"#{exp['id']} {exp.get('name', '')} [{exp['status']}]"
                self.exp_combo.addItem(label, exp["id"])

            restore_index = 0
            if current_id:
                for i in range(self.exp_combo.count()):
                    if self.exp_combo.itemData(i) == current_id:
                        restore_index = i
                        break
            self.exp_combo.setCurrentIndex(restore_index)
        finally:
            self.exp_combo.blockSignals(False)

    def _on_exp_changed(self, index):
        """切换实验"""
        if index < 0:
            return

        exp_id = self.exp_combo.itemData(index)
        if not exp_id:
            return

        exp = self.db.get_experiment_by_id(exp_id)
        if not exp:
            return

        # 更新信息面板
        self.exp_info_labels["状态"].setText(exp.get("status", "-"))
        type_map = {"distillation": "蒸馏", "pruning": "剪枝"}
        self.exp_info_labels["实验类型"].setText(type_map.get(exp["type"], exp["type"]))

        try:
            config = json.loads(exp["config"]) if isinstance(exp["config"], str) else exp["config"]
        except:
            config = {}

        # 判断蒸馏模式
        image_dir = config.get("image_dir", "").strip()
        is_multimodal = bool(image_dir) and exp["type"] == "distillation"
        self.exp_info_labels["蒸馏模式"].setText("多模态 (图文)" if is_multimodal else "纯文本")

        # 模型名称（提取路径末尾的模型名）
        def _extract_model_name(path):
            if not path:
                return "-"
            # 取路径最后一段，去掉可能的 snapshots/master 后缀
            parts = path.replace("\\", "/").split("/")
            name = parts[-1]
            if name in ("master", "snapshots"):
                name = parts[-2] if len(parts) >= 2 else "-"
            return name or "-"

        self.exp_info_labels["教师/目标模型"].setText(_extract_model_name(
            config.get("teacher_path", config.get("model_path", ""))
        ))
        self.exp_info_labels["学生模型"].setText(_extract_model_name(
            config.get("student_path", "")
        ))

        # 图像目录（多模态时显示）
        if is_multimodal:
            self.exp_info_labels["图像目录"].setText(image_dir)
        else:
            self.exp_info_labels["图像目录"].setText("-")

        # 蒸馏数据条数（结果里才有 num_data，config 里没有）
        try:
            eval_results = json.loads(exp.get("eval_results", "{}"))
        except (json.JSONDecodeError, TypeError):
            eval_results = {}
        num_data = eval_results.get("num_data", config.get("num_data", 0))
        num_images = eval_results.get("num_images", config.get("num_images", 0))
        if num_data:
            data_text = f"{num_data} 条"
            if num_images:
                data_text += f" / {num_images} 张图片"
            self.exp_info_labels["蒸馏数据"].setText(data_text)
        else:
            self.exp_info_labels["蒸馏数据"].setText("-")

        # 维度评分摘要（完整性/覆盖率/流畅度/一致性）
        scores = eval_results.get("scores", {})
        if scores and scores.get("dimensions"):
            dims = scores["dimensions"]
            b = scores.get("baseline", [])
            a = scores.get("after", [])
            parts = []
            for i, d in enumerate(dims):
                bv = b[i] if i < len(b) else 0
                av = a[i] if i < len(a) else 0
                arrow = "↑" if av > bv + 1 else ("↓" if av < bv - 1 else "→")
                parts.append(f"{d}: {bv:.0f}→{av:.0f}{arrow}")
            self.exp_info_labels["维度评分"].setText(" | ".join(parts))
        else:
            self.exp_info_labels["维度评分"].setText("-")

        self.exp_info_labels["开始时间"].setText(exp.get("started_at", "-"))
        self.exp_info_labels["完成时间"].setText(exp.get("completed_at", "-"))

        # 输出目录 / 报告
        output_path = exp.get("output_path", "")
        self.exp_info_labels["输出目录"].setText(output_path or "-")
        report_path = exp.get("report_path", "")
        self.open_report_btn.setEnabled(bool(report_path and os.path.exists(report_path)))
        self._current_report_path = report_path

        # 更新 Loss 曲线
        try:
            metrics = json.loads(exp.get("metrics", "[]"))
        except:
            metrics = []

        self.loss_chart_container.chart.clear_series()
        if metrics:
            for i, m in enumerate(metrics):
                self.loss_chart_container.chart.update_data(
                    "loss",
                    m.get("avg_loss", m.get("loss", 0)),
                    m.get("epoch", i)
                )

        baseline = eval_results.get("baseline", [])
        after = eval_results.get("after", [])
        teacher = eval_results.get("teacher", [])
        pruned = eval_results.get("pruned", [])
        recovered = eval_results.get("recovered", [])

        if exp["type"] == "distillation":
            after = after if after else [{}] * len(baseline)
            if teacher:
                # (教师, 蒸馏前, 蒸馏后)
                results_data = list(zip(teacher, baseline, after))
            else:
                results_data = list(zip([{}] * len(baseline), baseline, after))
        elif exp["type"] == "pruning":
            left = pruned if pruned else baseline
            results_data = list(zip(
                [{}] * len(left),
                left,
                recovered if recovered else [{}] * len(left),
            ))
        else:
            results_data = []

        # 根据是否有图片数据决定表格列数
        has_images = any(b.get("image") for b in baseline if b)
        show_teacher = exp["type"] == "distillation" and bool(teacher)
        headers = []
        if has_images:
            headers.append("图片")
        headers.append("问题")
        if show_teacher:
            headers.append("教师")
        headers += ["蒸馏前" if show_teacher else "基线", "蒸馏后" if show_teacher else "处理后", "对比分析"]
        self.eval_table.setColumnCount(len(headers))
        self.eval_table.setHorizontalHeaderLabels(headers)

        self.eval_table.setRowCount(len(results_data))
        for row, (t, b, a) in enumerate(results_data):
            col = 0
            if has_images:
                img_name = (t or b or {}).get("image", "-")
                self.eval_table.setItem(row, col, QTableWidgetItem(img_name))
                col += 1
            q = (t or b or {}).get("question", "")[:30]
            self.eval_table.setItem(row, col, QTableWidgetItem(q))
            col += 1
            if show_teacher:
                self.eval_table.setItem(row, col, QTableWidgetItem(t.get("answer", "")[:60] if t else "-"))
                col += 1
            self.eval_table.setItem(row, col, QTableWidgetItem(b.get("answer", "")[:60] if b else "-"))
            col += 1
            self.eval_table.setItem(row, col, QTableWidgetItem(a.get("answer", "")[:60] if a else "-"))
            col += 1
            if b and a and b.get("answer") == a.get("answer"):
                cmp = "相同"
            elif b and a:
                cmp = "不同"
            else:
                cmp = "-"
            self.eval_table.setItem(row, col, QTableWidgetItem(cmp))

        # 更新日志
        self.log_text.setText(f"实验 #{exp_id} 详情加载完成")

        # 失败实验显示删除按钮
        self.delete_btn.setVisible(exp.get("status") == "failed")

        # 逐图识别详情
        self._build_image_details(eval_results)

    def _build_image_details(self, eval_results):
        """按图片分组展示教师/蒸馏前/蒸馏后回答"""
        self.image_list.clear()
        self.image_detail_text.clear()
        self._image_groups = []
        baseline = eval_results.get("baseline", [])
        teacher = eval_results.get("teacher", [])
        after = eval_results.get("after", [])
        if not any(b.get("image") for b in baseline if b):
            return

        groups, order = {}, []
        for i, b in enumerate(baseline):
            key = b.get("image", "未知图片")
            if key not in groups:
                groups[key] = {"path": b.get("image_path", ""), "rows": []}
                order.append(key)
            t = teacher[i] if i < len(teacher) else {}
            a = after[i] if i < len(after) else {}
            groups[key]["rows"].append((b, t, a))

        for key in order:
            item = QListWidgetItem(key)
            path = groups[key]["path"]
            if path and os.path.isfile(path):
                reader = QImageReader(path)
                reader.setScaledSize(QSize(48, 48))
                pixmap = QPixmap.fromImage(reader.read())
                if not pixmap.isNull():
                    item.setIcon(QIcon(pixmap))
            self.image_list.addItem(item)
            self._image_groups.append({"name": key, **groups[key]})
        if self.image_list.count():
            self.image_list.setCurrentRow(0)
            self._on_image_selected(0)

    def _on_image_selected(self, row):
        """展示选中图片的逐题识别详情"""
        if row < 0 or row >= len(self._image_groups):
            return
        group = self._image_groups[row]
        lines = [f"图片: {group['name']}", "=" * 40]
        for b, t, a in group["rows"]:
            lines.append("")
            lines.append(f"【问题】{b.get('question', '')}")
            if t:
                lines.append(f"  教师  : {(t.get('answer') or '').strip()}")
            lines.append(f"  蒸馏前: {(b.get('answer') or '').strip()}")
            lines.append(f"  蒸馏后: {(a.get('answer') or '').strip()}")
        self.image_detail_text.setPlainText("\n".join(lines))

    def _open_report(self):
        """用系统默认应用打开评估报告"""
        path = getattr(self, "_current_report_path", "")
        if path and os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            QMessageBox.information(self, "提示", "评估报告不存在")

    def _delete_current(self):
        """删除当前实验"""
        exp_id = self.exp_combo.currentData()
        if not exp_id:
            return

        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除实验 #{exp_id} 吗？\n此操作不可恢复，将同时删除相关蒸馏数据。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.db.delete_experiment(exp_id)
        self.log_text.append(f"实验 #{exp_id} 已删除")
        self.refresh()

    def apply_theme(self):
        """应用当前主题"""
        t = Theme._colors[Theme.mode()]
        self.setStyleSheet("")
        self.title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {t['text_primary']};")
        self.exp_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {t["bg_secondary"]};
                color: {t["text_primary"]};
                border: 1px solid {t["border"]};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }}
            QComboBox:hover {{
                border-color: {t["accent"]};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
            QComboBox::down-arrow {{
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {t["text_secondary"]};
                margin-right: 6px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {t["bg_secondary"]};
                color: {t["text_primary"]};
                border: 1px solid {t["border"]};
                selection-background-color: {t["accent"]};
                selection-color: {t["bg_primary"]};
                padding: 4px;
            }}
        """)
        self.refresh_btn.setStyleSheet(f"""
            QPushButton {{ background-color: {t["btn_default"]}; color: {t["btn_default_text"]};
                padding: 6px 14px; border-radius: 4px; }}
            QPushButton:hover {{ background-color: {t["btn_default_hover"]}; }}
        """)
        self.open_report_btn.setStyleSheet(f"""
            QPushButton {{ background-color: {t["accent"]}; color: #ffffff;
                padding: 6px 14px; border-radius: 4px; font-weight: bold; }}
            QPushButton:hover {{ background-color: {t["accent_hover"]}; }}
            QPushButton:disabled {{ background-color: {t["bg_surface"]}; color: {t["text_muted"]}; }}
        """)
        for label in self.exp_info_labels.values():
            label.setStyleSheet(f"color: {t['text_primary']};")
        self.eval_table.setStyleSheet(f"""
            QTableWidget {{ background-color: {t["bg_secondary"]}; color: {t["text_primary"]};
                border: 1px solid {t["border"]}; gridline-color: {t["border"]}; }}
            QHeaderView::section {{ background-color: {t["header_bg"]}; color: {t["text_primary"]};
                padding: 4px; border: 1px solid {t["border"]}; font-weight: bold; }}
        """)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{ background-color: {t["bg_log"]}; color: {t["text_secondary"]};
                border: none; font-family: 'Consolas', monospace; font-size: 11px; }}
        """)
        self.image_list.setStyleSheet(f"""
            QListWidget {{ background-color: {t["bg_secondary"]}; color: {t["text_primary"]};
                border: 1px solid {t["border"]}; border-radius: 4px; font-size: 11px; }}
            QListWidget::item:selected {{ background-color: {t["bg_surface"]}; }}
        """)
        self.image_detail_text.setStyleSheet(f"""
            QTextEdit {{ background-color: {t["bg_log"]}; color: {t["text_secondary"]};
                border: 1px solid {t["border"]}; border-radius: 4px;
                font-family: 'Microsoft YaHei', sans-serif; font-size: 12px; }}
        """)
        self.loss_chart_container.apply_theme()
