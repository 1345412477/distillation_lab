"""剪枝实验页面 - 配置参数、启动剪枝、查看进度"""
import json
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
                              QCheckBox, QGroupBox, QFormLayout, QTextEdit,
                              QMessageBox, QFrame, QScrollArea, QProgressBar)
from PyQt6.QtCore import Qt, QSettings
from ..models.db_manager import DBManager
from ..components.loss_chart import LossChartWidget
from ..workers.train_worker import ExperimentWorker
from ..models.template_store import save_template, list_templates, load_template, delete_template
from ..utils.config import DEFAULT_PRUNE_CONFIG, EVAL_QUESTIONS
from ..utils.theme import Theme
from PyQt6.QtWidgets import QInputDialog


class PrunePage(QWidget):
    """剪枝实验配置与执行页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.db = DBManager()
        self.worker = None
        self.current_exp_id = None
        self._metrics = []
        self._form_labels = []
        self.init_ui()
        self._restore_session()

    def _restore_session(self):
        """恢复上次实验的参数配置"""
        try:
            raw = QSettings().value("prune/config", "")
            if raw:
                self._apply_template(json.loads(raw))
        except Exception:
            pass

    def _save_session(self):
        """保存当前参数配置（启动实验时调用）"""
        try:
            QSettings().setValue(
                "prune/config",
                json.dumps(self._read_config_values(), ensure_ascii=False),
            )
        except Exception:
            pass

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # 左侧：配置面板（紧凑布局，一屏装下）
        config_panel = QWidget()
        config_layout = QVBoxLayout(config_panel)
        config_layout.setSpacing(6)
        config_layout.setContentsMargins(0, 0, 0, 0)

        # 配置面板放入滚动区，避免小窗口下被截断
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setFrameShape(QFrame.Shape.NoFrame)
        config_scroll.setStyleSheet("QScrollArea { background: transparent; }")
        config_scroll.setWidget(config_panel)

        self.title = QLabel("剪枝实验配置")
        self.title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {Theme.get('text_primary')}; margin-bottom: 2px;")
        config_layout.addWidget(self.title)

        # 实验模板
        template_row = QHBoxLayout()
        template_row.addWidget(QLabel("模板:"))
        self.template_combo = QComboBox()
        self.template_combo.setMinimumWidth(140)
        template_row.addWidget(self.template_combo, 1)
        for text, handler in (("保存", self._save_template), ("加载", self._load_template),
                              ("删除", self._delete_template)):
            btn = QPushButton(text)
            btn.setMaximumWidth(44)
            btn.clicked.connect(handler)
            template_row.addWidget(btn)
        config_layout.addLayout(template_row)
        self.refresh_templates()

        # 模型选择
        model_group = QGroupBox("目标模型")
        model_layout = QFormLayout(model_group)
        model_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        model_layout.setContentsMargins(8, 4, 8, 4)
        model_layout.setSpacing(2)

        self.target_combo = QComboBox()
        self.target_combo.setMinimumWidth(180)
        self._add_row(model_layout, "目标模型:", self.target_combo)
        config_layout.addWidget(model_group)

        # 剪枝配置
        prune_group = QGroupBox("剪枝配置")
        prune_layout = QFormLayout(prune_group)
        prune_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        prune_layout.setContentsMargins(8, 4, 8, 4)
        prune_layout.setSpacing(2)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["FFN维度剪枝", "注意力头剪枝", "层剪枝"])
        self._add_row(prune_layout, "剪枝策略:", self.strategy_combo)

        self.ratio_spin = QSpinBox()
        self.ratio_spin.setRange(5, 50)
        self.ratio_spin.setValue(int(DEFAULT_PRUNE_CONFIG["prune_ratio"] * 100))
        self.ratio_spin.setSuffix(" %")
        self.ratio_spin.setToolTip("建议从 10-20% 开始尝试，避免性能严重下降")
        self._add_row(prune_layout, "剪枝比例:", self.ratio_spin)

        self.importance_combo = QComboBox()
        self.importance_combo.addItems(["L1范数", "L2范数"])
        self._add_row(prune_layout, "重要性评估:", self.importance_combo)

        config_layout.addWidget(prune_group)

        # 恢复配置
        recovery_group = QGroupBox("恢复微调配置")
        recovery_layout = QFormLayout(recovery_group)
        recovery_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        recovery_layout.setContentsMargins(8, 4, 8, 4)
        recovery_layout.setSpacing(2)

        self.enable_recovery_cb = QCheckBox("启用恢复微调")
        self.enable_recovery_cb.setChecked(DEFAULT_PRUNE_CONFIG["enable_recovery"])
        self.enable_recovery_cb.setStyleSheet(f"color: {Theme.get('text_primary')}; font-size: 12px;")
        recovery_layout.addRow(self.enable_recovery_cb)

        self.recovery_epochs_spin = QSpinBox()
        self.recovery_epochs_spin.setRange(1, 20)
        self.recovery_epochs_spin.setValue(DEFAULT_PRUNE_CONFIG["recovery_epochs"])
        self.recovery_epochs_spin.setToolTip("增加恢复轮数以改善剪枝后性能")
        self._add_row(recovery_layout, "恢复轮数:", self.recovery_epochs_spin)

        self.recovery_lr_spin = QDoubleSpinBox()
        self.recovery_lr_spin.setRange(1e-6, 1e-2)
        self.recovery_lr_spin.setDecimals(6)
        self.recovery_lr_spin.setSingleStep(1e-5)
        self.recovery_lr_spin.setValue(DEFAULT_PRUNE_CONFIG["recovery_lr"])
        self._add_row(recovery_layout, "恢复学习率:", self.recovery_lr_spin)

        self.recovery_lora_r_spin = QSpinBox()
        self.recovery_lora_r_spin.setRange(4, 64)
        self.recovery_lora_r_spin.setValue(DEFAULT_PRUNE_CONFIG["recovery_lora_r"])
        self._add_row(recovery_layout, "恢复LoRA Rank:", self.recovery_lora_r_spin)

        self.recovery_lora_alpha_spin = QSpinBox()
        self.recovery_lora_alpha_spin.setRange(8, 128)
        self.recovery_lora_alpha_spin.setValue(DEFAULT_PRUNE_CONFIG["recovery_lora_alpha"])
        self._add_row(recovery_layout, "恢复LoRA Alpha:", self.recovery_lora_alpha_spin)

        self.recovery_lora_dropout_spin = QDoubleSpinBox()
        self.recovery_lora_dropout_spin.setRange(0.0, 0.5)
        self.recovery_lora_dropout_spin.setSingleStep(0.05)
        self.recovery_lora_dropout_spin.setValue(DEFAULT_PRUNE_CONFIG["recovery_lora_dropout"])
        self._add_row(recovery_layout, "恢复LoRA Dropout:", self.recovery_lora_dropout_spin)

        config_layout.addWidget(recovery_group)

        # 评估问题自定义（紧凑）
        eval_group = QGroupBox("评估问题设置（每行一个问题）")
        eval_layout = QVBoxLayout(eval_group)
        eval_layout.setContentsMargins(8, 4, 8, 4)
        eval_layout.setSpacing(2)
        eval_hint = QLabel("将使用这些问题对剪枝前后的模型进行评估对比")
        eval_hint.setStyleSheet(f"color: {Theme.get('text_muted')}; font-size: 10px;")
        eval_layout.addWidget(eval_hint)

        self.eval_questions_edit = QTextEdit()
        self.eval_questions_edit.setPlaceholderText("每行输入一个评估问题...")
        self.eval_questions_edit.setMinimumHeight(48)
        self.eval_questions_edit.setMaximumHeight(60)
        self.eval_questions_edit.setPlainText("\n".join(EVAL_QUESTIONS))
        self.eval_questions_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Theme.get("bg_secondary")};
                color: {Theme.get("text_primary")};
                border: 1px solid {Theme.get("border")};
                border-radius: 3px;
                padding: 3px;
                font-size: 11px;
            }}
        """)
        eval_layout.addWidget(self.eval_questions_edit)

        eval_tokens_row = QHBoxLayout()
        eval_tokens_row.addWidget(QLabel("评估最大Token:"))
        self.eval_max_tokens_spin = QSpinBox()
        self.eval_max_tokens_spin.setRange(16, 512)
        self.eval_max_tokens_spin.setValue(DEFAULT_PRUNE_CONFIG["eval_max_new_tokens"])
        eval_tokens_row.addWidget(self.eval_max_tokens_spin)
        eval_tokens_row.addStretch()
        eval_layout.addLayout(eval_tokens_row)
        config_layout.addWidget(eval_group)

        # 参考答案（可选，用于自动评分）
        ref_group = QGroupBox("参考答案（可选，每行一条，与评估问题一一对应）")
        ref_layout = QVBoxLayout(ref_group)
        ref_layout.setContentsMargins(8, 4, 8, 4)
        ref_layout.setSpacing(2)
        ref_hint = QLabel("填写后将对回答进行“内容覆盖率”自动评分；留空则只评估完整性与流畅度")
        ref_hint.setStyleSheet(f"color: {Theme.get('text_muted')}; font-size: 10px;")
        ref_layout.addWidget(ref_hint)
        self.reference_answers_edit = QTextEdit()
        self.reference_answers_edit.setPlaceholderText("每行输入一个参考答案...")
        self.reference_answers_edit.setMinimumHeight(40)
        self.reference_answers_edit.setMaximumHeight(56)
        self.reference_answers_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Theme.get("bg_secondary")};
                color: {Theme.get("text_primary")};
                border: 1px solid {Theme.get("border")};
                border-radius: 3px;
                padding: 3px;
                font-size: 11px;
            }}
        """)
        ref_layout.addWidget(self.reference_answers_edit)
        config_layout.addWidget(ref_group)

        # 操作按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ 开始剪枝")
        self.start_btn.clicked.connect(self._start_experiment)
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.get("yellow")}; color: #ffffff;
                padding: 6px 16px; border-radius: 4px; font-weight: bold;
                font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {Theme.get("peach")}; }}
        """)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■ 停止")
        self.stop_btn.clicked.connect(self._stop_experiment)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.get("red")}; color: #ffffff;
                padding: 6px 16px; border-radius: 4px; font-weight: bold;
                font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {Theme.get("red")}; }}
        """)
        btn_layout.addWidget(self.stop_btn)
        config_layout.addLayout(btn_layout)

        config_layout.addStretch()

        # 右侧：监控面板
        monitor_panel = QWidget()
        monitor_layout = QVBoxLayout(monitor_panel)
        monitor_layout.setSpacing(4)
        monitor_layout.setContentsMargins(0, 0, 0, 0)

        self.monitor_title = QLabel("实验状态")
        self.monitor_title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {Theme.get('text_primary')}; margin-bottom: 2px;")
        monitor_layout.addWidget(self.monitor_title)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet(f"color: {Theme.get('text_secondary')}; font-size: 12px; padding: 2px;")
        monitor_layout.addWidget(self.status_label)

        # 阶段进度
        self.phase_label = QLabel("阶段: -")
        self.phase_label.setStyleSheet(f"color: {Theme.get('text_secondary')}; font-size: 11px;")
        monitor_layout.addWidget(self.phase_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        monitor_layout.addWidget(self.progress_bar)

        self.loss_chart = LossChartWidget(title="恢复微调 Loss")
        self.loss_chart.setMinimumHeight(180)
        monitor_layout.addWidget(self.loss_chart)

        self.log_label = QLabel("实时日志")
        self.log_label.setStyleSheet(f"font-weight: bold; color: {Theme.get('text_primary')}; font-size: 12px;")
        monitor_layout.addWidget(self.log_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Theme.get("bg_log")}; color: {Theme.get("text_secondary")};
                border: 1px solid {Theme.get("border")}; border-radius: 4px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
            }}
        """)
        monitor_layout.addWidget(self.log_text)

        main_layout.addWidget(config_scroll, 2)
        main_layout.addWidget(monitor_panel, 3)

    def _add_row(self, layout, label_text, widget):
        """添加表单行，确保标签颜色正确"""
        label = QLabel(label_text)
        label.setStyleSheet(f"color: {Theme.get('text_primary')}; font-size: 12px; font-weight: 500;")
        layout.addRow(label, widget)
        self._form_labels.append(label)

    def refresh_models(self):
        """刷新模型下拉列表"""
        models = self.db.get_all_models()
        self.target_combo.clear()
        for m in models:
            text = f"{m['name']} ({m.get('parameters', 0):.3f}B)"
            self.target_combo.addItem(text, m["id"])

    def _read_config_values(self) -> dict:
        """读取全部可调参数（不含模型选择），供启动实验与模板复用"""
        return {
            "prune_strategy": ["ffn", "attention", "layer"][self.strategy_combo.currentIndex()],
            "prune_ratio": self.ratio_spin.value() / 100.0,
            "importance_metric": ["l1", "l2"][self.importance_combo.currentIndex()],
            "enable_recovery": self.enable_recovery_cb.isChecked(),
            "recovery_epochs": self.recovery_epochs_spin.value(),
            "recovery_lr": self.recovery_lr_spin.value(),
            "recovery_lora_r": self.recovery_lora_r_spin.value(),
            "recovery_lora_alpha": self.recovery_lora_alpha_spin.value(),
            "recovery_lora_dropout": self.recovery_lora_dropout_spin.value(),
            "reference_answers": [r.strip() for r in self.reference_answers_edit.toPlainText().strip().split("\n") if r.strip()],
            "eval_questions": [q.strip() for q in self.eval_questions_edit.toPlainText().strip().split("\n") if q.strip()],
            "eval_max_new_tokens": self.eval_max_tokens_spin.value(),
            "seed": DEFAULT_PRUNE_CONFIG["seed"],
        }

    def _get_config(self):
        """获取完整实验配置（含目标模型）"""
        target_idx = self.target_combo.currentIndex()
        if target_idx < 0:
            return None

        target_id = self.target_combo.itemData(target_idx)
        target_model = self.db.get_model_by_id(target_id)

        config = self._read_config_values()
        config.update({
            "target_model_id": target_id,
            "model_path": target_model["local_path"] if target_model else "",
        })
        return config

    # ---- 实验模板 ----

    def refresh_templates(self):
        current = self.template_combo.currentText()
        self.template_combo.clear()
        for tpl in list_templates():
            self.template_combo.addItem(tpl["name"])
        if current:
            idx = self.template_combo.findText(current)
            if idx >= 0:
                self.template_combo.setCurrentIndex(idx)

    def _save_template(self, *args):
        name, ok = QInputDialog.getText(self, "保存实验模板", "模板名称：")
        if not ok or not name.strip():
            return
        saved = save_template(name, self._read_config_values())
        self.refresh_templates()
        idx = self.template_combo.findText(saved)
        if idx >= 0:
            self.template_combo.setCurrentIndex(idx)
        self._on_log(f"模板已保存: {saved}")

    def _load_template(self, *args):
        name = self.template_combo.currentText()
        if not name:
            QMessageBox.information(self, "提示", "请先选择要加载的模板")
            return
        try:
            cfg = load_template(name)
        except FileNotFoundError as e:
            QMessageBox.warning(self, "提示", str(e))
            return
        self._apply_template(cfg)
        self._on_log(f"已加载模板: {name}")

    def _delete_template(self, *args):
        name = self.template_combo.currentText()
        if not name:
            return
        reply = QMessageBox.question(
            self, "确认删除", f"确定要删除模板 [{name}] 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            delete_template(name)
            self.refresh_templates()
            self._on_log(f"模板已删除: {name}")

    def _apply_template(self, cfg: dict):
        self.strategy_combo.setCurrentIndex(
            {"ffn": 0, "attention": 1, "layer": 2}.get(cfg.get("prune_strategy", "ffn"), 0)
        )
        self.ratio_spin.setValue(int(round(cfg.get("prune_ratio", 0.2) * 100)))
        self.importance_combo.setCurrentIndex(
            {"l1": 0, "l2": 1}.get(cfg.get("importance_metric", "l1"), 0)
        )
        self.enable_recovery_cb.setChecked(bool(cfg.get("enable_recovery", True)))
        self.recovery_epochs_spin.setValue(int(cfg.get("recovery_epochs", 5)))
        self.recovery_lr_spin.setValue(float(cfg.get("recovery_lr", 5e-5)))
        self.recovery_lora_r_spin.setValue(int(cfg.get("recovery_lora_r", 16)))
        self.recovery_lora_alpha_spin.setValue(int(cfg.get("recovery_lora_alpha", 32)))
        self.recovery_lora_dropout_spin.setValue(float(cfg.get("recovery_lora_dropout", 0.05)))
        self.eval_max_tokens_spin.setValue(int(cfg.get("eval_max_new_tokens", 64)))
        if cfg.get("eval_questions"):
            self.eval_questions_edit.setPlainText("\n".join(cfg["eval_questions"]))
        if cfg.get("reference_answers"):
            self.reference_answers_edit.setPlainText("\n".join(cfg["reference_answers"]))

    def _start_experiment(self):
        config = self._get_config()
        if not config:
            QMessageBox.warning(self, "提示", "请先导入目标模型")
            return

        self._save_session()

        prune_pct = config["prune_ratio"] * 100
        if prune_pct > 30:
            reply = QMessageBox.question(
                self, "确认剪枝比例",
                f"剪枝比例 {prune_pct:.0f}% 较高，可能导致模型性能严重下降。\n建议从 10-20% 开始尝试。\n\n是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        target_model = self.db.get_model_by_id(config["target_model_id"])
        exp_name = f"剪枝 {target_model['name']} ({prune_pct:.0f}%)"

        exp_id = self.db.add_experiment(
            name=exp_name,
            exp_type="pruning",
            config=config,
            student_model_id=config["target_model_id"],
        )
        config["exp_id"] = exp_id
        self.current_exp_id = exp_id

        self.worker = ExperimentWorker("pruning", config)
        self.worker.status_updated.connect(self._on_status)
        self.worker.metric_updated.connect(self._on_metric)
        self.worker.log_updated.connect(self._on_log)
        self.worker.phase_updated.connect(self._on_phase)
        self.worker.baseline_ready.connect(lambda d: self._on_log("基线评估完成"))
        self.worker.prune_stats_ready.connect(self._on_prune_stats)
        self.worker.importance_distribution.connect(lambda d: self._on_log("重要性分析完成"))
        self.worker.experiment_completed.connect(self._on_complete)
        self.worker.experiment_error.connect(self._on_error)

        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.loss_chart.clear_series()
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.phase_label.setText("阶段: -")
        self._on_log(f"实验 #{exp_id} 启动: {exp_name}")
        self.db.update_experiment_status(exp_id, "running")

    def _stop_experiment(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self._on_log("正在停止实验，等待当前步骤完成后退出...")

    def stop(self):
        """应用关闭时调用：请求停止实验（协作式）"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def _on_status(self, msg):
        self.status_label.setText(f"状态: {msg}")

    def _on_metric(self, name, value, step):
        if name == "recovery_loss":
            self.loss_chart.update_data("recovery_loss", value, step)
            self._metrics.append({"epoch": step, "loss": value})

    def _on_log(self, msg):
        self.log_text.append(msg)

    def _on_phase(self, phase, current, total):
        self.phase_label.setText(f"阶段: {phase}")
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(min(current, total))
            pct = int(current / total * 100)
            self.progress_bar.setFormat(f"{phase} {current}/{total} ({pct}%)")

    def _on_prune_stats(self, stats):
        if stats.get("mode") == "rebuild":
            before = stats.get("parameters_before", 0)
            after = stats.get("parameters_after", 0)
            pct = stats.get("param_reduction_ratio", 0) * 100
            self._on_log(
                f"剪枝统计: 参数 {before/1e6:.1f}M → {after/1e6:.1f}M（缩减 {pct:.1f}%）"
            )
        else:
            ratio = stats.get("zero_ratio", 0) * 100
            self._on_log(f"剪枝统计(置零回退): 零参数占比 {ratio:.1f}%")

    def _on_complete(self, results):
        self._on_log("实验完成！")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("状态: 完成")

        if self.current_exp_id:
            self._metrics = []
            self._on_log(f"结果已保存（输出目录: {results.get('output_path', '-')}）")

    def _on_error(self, err_msg):
        self._on_log(f"错误: {err_msg}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("状态: 失败")

    def apply_theme(self):
        """应用当前主题"""
        t = Theme._colors[Theme.mode()]
        self.setStyleSheet("")
        self.title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {t['text_primary']}; margin-bottom: 2px;")
        self.monitor_title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {t['text_primary']}; margin-bottom: 2px;")
        self.status_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 12px; padding: 2px;")
        self.phase_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 11px;")
        self.log_label.setStyleSheet(f"font-weight: bold; color: {t['text_primary']}; font-size: 12px;")
        for edit in (self.eval_questions_edit, self.reference_answers_edit):
            edit.setStyleSheet(f"""
                QTextEdit {{
                    background-color: {t["bg_secondary"]};
                    color: {t["text_primary"]};
                    border: 1px solid {t["border"]};
                    border-radius: 3px;
                    padding: 3px;
                    font-size: 11px;
                }}
            """)
        # 更新表单标签颜色
        for label in self._form_labels:
            label.setStyleSheet(f"color: {t['text_primary']}; font-size: 12px; font-weight: 500;")
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {t["yellow"]}; color: #ffffff;
                padding: 6px 16px; border-radius: 4px; font-weight: bold;
                font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {t["peach"]}; }}
        """)
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {t["red"]}; color: #ffffff;
                padding: 6px 16px; border-radius: 4px; font-weight: bold;
                font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {t["red"]}; }}
        """)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {t["bg_log"]}; color: {t["text_secondary"]};
                border: 1px solid {t["border"]}; border-radius: 4px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
            }}
        """)
        self.loss_chart.apply_theme()
