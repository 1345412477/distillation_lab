"""蒸馏实验页面 - 配置参数、启动蒸馏、查看进度"""
import json
import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
                              QCheckBox, QGroupBox, QFormLayout, QTextEdit,
                              QSplitter, QFrame, QMessageBox,
                              QLineEdit, QFileDialog, QScrollArea, QListWidget,
                              QProgressBar, QListWidgetItem)
from PyQt6.QtCore import Qt, QSize, QSettings
from PyQt6.QtGui import QImageReader, QPixmap, QIcon
from ..models.db_manager import DBManager
from ..components.loss_chart import LossChartWidget
from ..components.collapsible_section import CollapsibleSection
from ..workers.train_worker import ExperimentWorker, BatchWorker
from ..models.template_store import save_template, list_templates, load_template, delete_template
from ..utils.config import (DEFAULT_DISTILL_CONFIG, EVAL_QUESTIONS,
                            DEFAULT_IMAGE_CONFIG, IMAGE_EVAL_QUESTIONS, SEED_QUESTIONS)
from ..utils.theme import Theme
from PyQt6.QtWidgets import QInputDialog, QDialog, QListWidget, QListWidgetItem


class DistillPage(QWidget):
    """蒸馏实验配置与执行页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.db = DBManager()
        self.worker = None
        self._batch = None
        self.current_exp_id = None
        self._metrics = []
        self._distill_data = []
        self._form_labels = []
        self._sections = []
        self._phase_total = 0
        self.init_ui()
        self._restore_session()

    def _restore_session(self):
        """恢复上次实验的参数配置"""
        try:
            raw = QSettings().value("distill/config", "")
            if raw:
                self._apply_template(json.loads(raw))
        except Exception:
            pass

    def _save_session(self):
        """保存当前参数配置（启动实验/批量时调用）"""
        try:
            QSettings().setValue(
                "distill/config",
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

        self.title = QLabel("蒸馏实验配置")
        self.title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {Theme.get('text_primary')}; margin-bottom: 2px;")
        config_layout.addWidget(self.title)

        # 实验模板（参数组合保存/复用，便于批量调参）
        template_row = QHBoxLayout()
        template_row.addWidget(QLabel("模板:"))
        self.template_combo = QComboBox()
        self.template_combo.setMinimumWidth(140)
        template_row.addWidget(self.template_combo, 1)
        for text, handler in (("保存", self._save_template), ("加载", self._load_template),
                              ("删除", self._delete_template), ("批量", self._start_batch)):
            btn = QPushButton(text)
            btn.setMaximumWidth(40 if text != "批量" else 44)
            btn.clicked.connect(handler)
            template_row.addWidget(btn)
        config_layout.addLayout(template_row)
        self.refresh_templates()

        # 模型选择
        model_group = QGroupBox("模型选择")
        model_layout = QFormLayout(model_group)
        model_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        model_layout.setContentsMargins(8, 4, 8, 4)
        model_layout.setSpacing(2)

        self.teacher_combo = QComboBox()
        self.teacher_combo.setMinimumWidth(180)
        self.student_combo = QComboBox()
        self.student_combo.setMinimumWidth(180)
        self._add_row(model_layout, "教师模型:", self.teacher_combo)
        self._add_row(model_layout, "学生模型:", self.student_combo)
        config_layout.addWidget(model_group)

        # 数据生成配置
        data_group = QGroupBox("数据生成配置")
        data_layout = QFormLayout(data_group)
        data_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        data_layout.setContentsMargins(8, 4, 8, 4)
        data_layout.setSpacing(2)

        self.num_generate_spin = QSpinBox()
        self.num_generate_spin.setRange(5, 500)
        self.num_generate_spin.setValue(DEFAULT_DISTILL_CONFIG["num_generate"])
        self._add_row(data_layout, "生成数量:", self.num_generate_spin)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.1, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(DEFAULT_DISTILL_CONFIG["temperature"])
        self._add_row(data_layout, "温度:", self.temp_spin)

        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.1, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setValue(DEFAULT_DISTILL_CONFIG["top_p"])
        self._add_row(data_layout, "Top-P:", self.top_p_spin)

        self.max_token_spin = QSpinBox()
        self.max_token_spin.setRange(64, 1024)
        self.max_token_spin.setValue(DEFAULT_DISTILL_CONFIG["max_new_tokens"])
        self._add_row(data_layout, "最大Token:", self.max_token_spin)

        config_layout.addWidget(data_group)

        # 高级生成参数（默认折叠，保持界面简洁）
        data_adv = CollapsibleSection("高级生成参数（Top-K / Beam / 重复惩罚 / 思考模式）")
        self._sections.append(data_adv)
        adv_data_widget = QWidget()
        adv_data_form = QFormLayout(adv_data_widget)
        adv_data_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        adv_data_form.setContentsMargins(0, 0, 0, 0)
        adv_data_form.setSpacing(2)

        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(0, 100)
        self.top_k_spin.setValue(DEFAULT_DISTILL_CONFIG["top_k"])
        self.top_k_spin.setToolTip("0 = 不限制")
        self._add_row(adv_data_form, "Top-K:", self.top_k_spin)

        self.rep_penalty_spin = QDoubleSpinBox()
        self.rep_penalty_spin.setRange(1.0, 2.0)
        self.rep_penalty_spin.setSingleStep(0.1)
        self.rep_penalty_spin.setValue(DEFAULT_DISTILL_CONFIG["repetition_penalty"])
        self._add_row(adv_data_form, "重复惩罚:", self.rep_penalty_spin)

        self.num_beams_spin = QSpinBox()
        self.num_beams_spin.setRange(1, 8)
        self.num_beams_spin.setValue(DEFAULT_DISTILL_CONFIG["num_beams"])
        self.num_beams_spin.setToolTip(">1 时使用束搜索（自动关闭采样）")
        self._add_row(adv_data_form, "Beam数:", self.num_beams_spin)

        self.enable_think_cb = QCheckBox("启用思考模式（Qwen enable_thinking）")
        self.enable_think_cb.setChecked(DEFAULT_DISTILL_CONFIG["enable_think"])
        self.enable_think_cb.setStyleSheet(f"color: {Theme.get('text_primary')}; font-size: 12px;")
        adv_data_form.addRow("", self.enable_think_cb)
        data_adv.add_widget(adv_data_widget)
        config_layout.addWidget(data_adv)

        # 图像数据集配置（多模态蒸馏）
        image_group = QGroupBox("图像数据集配置（多模态蒸馏）")
        image_group.setToolTip("留空则执行纯文本蒸馏。填入图像目录后，将进行多模态蒸馏")
        image_layout = QFormLayout(image_group)
        image_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        image_layout.setContentsMargins(8, 4, 8, 4)
        image_layout.setSpacing(2)

        self.image_dir_input = QLineEdit()
        self.image_dir_input.setPlaceholderText("留空=纯文本蒸馏，选择包含图片的目录=多模态蒸馏")
        self.image_dir_input.setMinimumWidth(180)

        image_dir_layout = QHBoxLayout()
        image_dir_layout.setContentsMargins(0, 0, 0, 0)
        image_dir_layout.addWidget(self.image_dir_input)

        browse_image_btn = QPushButton("浏览...")
        browse_image_btn.clicked.connect(self._browse_image_dir)
        browse_image_btn.setStyleSheet("max-width: 55px; padding: 2px 6px;")
        image_dir_layout.addWidget(browse_image_btn)

        self.image_count_label = QLabel("未选择目录")
        self.image_count_label.setStyleSheet(f"color: {Theme.get('text_muted')}; font-size: 11px;")
        image_dir_layout.addWidget(self.image_count_label)
        image_dir_layout.addStretch()

        image_layout.addRow("图像目录:", image_dir_layout)
        self._form_labels.append(QLabel("图像目录:"))

        self.image_prompts_edit = QTextEdit()
        self.image_prompts_edit.setPlainText("\n".join(DEFAULT_IMAGE_CONFIG["image_prompts"]))
        self.image_prompts_edit.setPlaceholderText("每行一个提示词；多行 = 每张图片生成多条数据（数据增强）")
        self.image_prompts_edit.setMinimumHeight(46)
        self.image_prompts_edit.setMaximumHeight(64)
        self.image_prompts_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Theme.get("bg_secondary")};
                color: {Theme.get("text_primary")};
                border: 1px solid {Theme.get("border")};
                border-radius: 3px;
                padding: 3px;
                font-size: 11px;
            }}
        """)
        image_layout.addRow("图片提示词:", self.image_prompts_edit)

        self.max_images_spin = QSpinBox()
        self.max_images_spin.setRange(1, 200)
        self.max_images_spin.setValue(DEFAULT_IMAGE_CONFIG["max_images"])
        self._add_row(image_layout, "最大图片数:", self.max_images_spin)

        config_layout.addWidget(image_group)

        # 评估问题自定义（紧凑）
        eval_group = QGroupBox("评估问题设置（每行一个问题）")
        eval_layout = QVBoxLayout(eval_group)
        eval_layout.setContentsMargins(8, 4, 8, 4)
        eval_layout.setSpacing(2)
        eval_hint = QLabel("将使用这些问题对蒸馏前后的模型进行评估对比")
        eval_hint.setStyleSheet(f"color: {Theme.get('text_muted')}; font-size: 10px;")
        eval_layout.addWidget(eval_hint)

        self.eval_questions_edit = QTextEdit()
        self.eval_questions_edit.setPlaceholderText("每行输入一个评估问题...")
        self.eval_questions_edit.setMinimumHeight(48)
        self.eval_questions_edit.setMaximumHeight(60)
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
        self._init_eval_questions()
        eval_layout.addWidget(self.eval_questions_edit)

        eval_tokens_row = QHBoxLayout()
        eval_tokens_row.addWidget(QLabel("评估最大Token:"))
        self.eval_max_tokens_spin = QSpinBox()
        self.eval_max_tokens_spin.setRange(16, 512)
        self.eval_max_tokens_spin.setValue(DEFAULT_DISTILL_CONFIG["eval_max_new_tokens"])
        eval_tokens_row.addWidget(self.eval_max_tokens_spin)
        eval_tokens_row.addStretch()
        eval_layout.addLayout(eval_tokens_row)

        self.use_clip_cb = QCheckBox("CLIP 图像对齐评分（可选，衡量描述与图片的语义匹配，首次需下载模型）")
        self.use_clip_cb.setChecked(False)
        self.use_clip_cb.setStyleSheet(f"color: {Theme.get('text_primary')}; font-size: 12px;")
        eval_layout.addWidget(self.use_clip_cb)
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

        # 种子问题（默认折叠，教师据此生成数据）
        seed_adv = CollapsibleSection("种子问题（教师生成数据用，每行一个）")
        self._sections.append(seed_adv)
        seed_widget = QWidget()
        seed_layout = QVBoxLayout(seed_widget)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_layout.setSpacing(2)
        self.seed_questions_edit = QTextEdit()
        self.seed_questions_edit.setPlainText("\n".join(SEED_QUESTIONS))
        self.seed_questions_edit.setMinimumHeight(48)
        self.seed_questions_edit.setMaximumHeight(80)
        self.seed_questions_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Theme.get("bg_secondary")};
                color: {Theme.get("text_primary")};
                border: 1px solid {Theme.get("border")};
                border-radius: 3px;
                padding: 3px;
                font-size: 11px;
            }}
        """)
        seed_layout.addWidget(self.seed_questions_edit)
        seed_adv.add_widget(seed_widget)
        config_layout.addWidget(seed_adv)

        # 训练配置
        train_group = QGroupBox("训练配置")
        train_layout = QFormLayout(train_group)
        train_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        train_layout.setContentsMargins(8, 4, 8, 4)
        train_layout.setSpacing(2)

        self.lora_r_spin = QSpinBox()
        self.lora_r_spin.setRange(4, 64)
        self.lora_r_spin.setValue(DEFAULT_DISTILL_CONFIG["lora_r"])
        self._add_row(train_layout, "LoRA Rank:", self.lora_r_spin)

        self.lora_alpha_spin = QSpinBox()
        self.lora_alpha_spin.setRange(8, 128)
        self.lora_alpha_spin.setValue(DEFAULT_DISTILL_CONFIG["lora_alpha"])
        self._add_row(train_layout, "LoRA Alpha:", self.lora_alpha_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10)
        self.epochs_spin.setValue(DEFAULT_DISTILL_CONFIG["epochs"])
        self._add_row(train_layout, "训练轮数:", self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1e-2)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(1e-5)
        self.lr_spin.setValue(DEFAULT_DISTILL_CONFIG["learning_rate"])
        self._add_row(train_layout, "学习率:", self.lr_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 8)
        self.batch_spin.setValue(DEFAULT_DISTILL_CONFIG["batch_size"])
        self._add_row(train_layout, "批大小:", self.batch_spin)

        self.use_4bit_cb = QCheckBox("4-bit 量化（节省显存，适合 6GB 以下 GPU）")
        self.use_4bit_cb.setChecked(True)
        self.use_4bit_cb.setStyleSheet(f"color: {Theme.get('text_primary')}; font-size: 12px;")
        train_layout.addRow("", self.use_4bit_cb)

        config_layout.addWidget(train_group)

        # 高级训练参数（默认折叠）
        train_adv = CollapsibleSection("高级训练参数（优化器 / 正则 / 种子 / 4-bit 细节）")
        self._sections.append(train_adv)
        adv_train_widget = QWidget()
        adv_train_form = QFormLayout(adv_train_widget)
        adv_train_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        adv_train_form.setContentsMargins(0, 0, 0, 0)
        adv_train_form.setSpacing(2)

        self.lora_dropout_spin = QDoubleSpinBox()
        self.lora_dropout_spin.setRange(0.0, 0.5)
        self.lora_dropout_spin.setSingleStep(0.05)
        self.lora_dropout_spin.setValue(DEFAULT_DISTILL_CONFIG["lora_dropout"])
        self._add_row(adv_train_form, "LoRA Dropout:", self.lora_dropout_spin)

        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.5)
        self.weight_decay_spin.setDecimals(3)
        self.weight_decay_spin.setSingleStep(0.005)
        self.weight_decay_spin.setValue(DEFAULT_DISTILL_CONFIG["weight_decay"])
        self._add_row(adv_train_form, "权重衰减:", self.weight_decay_spin)

        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 200)
        self.warmup_spin.setValue(DEFAULT_DISTILL_CONFIG["warmup_steps"])
        self._add_row(adv_train_form, "Warmup步数:", self.warmup_spin)

        self.grad_accum_spin = QSpinBox()
        self.grad_accum_spin.setRange(1, 32)
        self.grad_accum_spin.setValue(DEFAULT_DISTILL_CONFIG["grad_accum_steps"])
        self._add_row(adv_train_form, "梯度累积:", self.grad_accum_spin)

        self.max_seq_len_spin = QSpinBox()
        self.max_seq_len_spin.setRange(64, 2048)
        self.max_seq_len_spin.setSingleStep(64)
        self.max_seq_len_spin.setValue(DEFAULT_DISTILL_CONFIG["max_seq_len"])
        self._add_row(adv_train_form, "最大序列长度:", self.max_seq_len_spin)

        self.grad_clip_spin = QDoubleSpinBox()
        self.grad_clip_spin.setRange(0.1, 10.0)
        self.grad_clip_spin.setSingleStep(0.1)
        self.grad_clip_spin.setValue(DEFAULT_DISTILL_CONFIG["grad_clip"])
        self._add_row(adv_train_form, "梯度裁剪:", self.grad_clip_spin)

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["AdamW", "AdamW8bit", "SGD"])
        self._add_row(adv_train_form, "优化器:", self.optimizer_combo)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(DEFAULT_DISTILL_CONFIG["seed"])
        self._add_row(adv_train_form, "随机种子:", self.seed_spin)

        self.quant_type_combo = QComboBox()
        self.quant_type_combo.addItems(["nf4", "fp4"])
        self.quant_type_combo.setCurrentText("nf4")
        self._add_row(adv_train_form, "量化类型:", self.quant_type_combo)

        self.double_quant_cb = QCheckBox("双量化（Double Quant）")
        self.double_quant_cb.setChecked(True)
        self.double_quant_cb.setStyleSheet(f"color: {Theme.get('text_primary')}; font-size: 12px;")
        adv_train_form.addRow("", self.double_quant_cb)

        self.compute_dtype_combo = QComboBox()
        self.compute_dtype_combo.addItems(["fp16", "bf16"])
        self.compute_dtype_combo.setCurrentText("fp16")
        self._add_row(adv_train_form, "计算精度:", self.compute_dtype_combo)
        train_adv.add_widget(adv_train_widget)
        config_layout.addWidget(train_adv)

        # 操作按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ 开始蒸馏")
        self.start_btn.clicked.connect(self._start_experiment)
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.get("green")}; color: #ffffff;
                padding: 6px 16px; border-radius: 4px; font-weight: bold;
                font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {Theme.get("teal")}; }}
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

        # 状态标签
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
        self.progress_bar.setFormat("%v/%m")
        monitor_layout.addWidget(self.progress_bar)

        # 速度 / 耗时 / 预计剩余
        stats_row = QHBoxLayout()
        self.speed_label = QLabel("速度: -")
        self.elapsed_label = QLabel("耗时: -")
        self.eta_label = QLabel("剩余: -")
        for lb in (self.speed_label, self.elapsed_label, self.eta_label):
            lb.setStyleSheet(f"color: {Theme.get('text_muted')}; font-size: 11px;")
            stats_row.addWidget(lb)
        stats_row.addStretch()
        monitor_layout.addLayout(stats_row)

        # 逐样本识别结果预览
        self.sample_label = QLabel("识别结果预览")
        self.sample_label.setStyleSheet(f"color: {Theme.get('text_muted')}; font-size: 11px;")
        monitor_layout.addWidget(self.sample_label)
        self.sample_list = QListWidget()
        self.sample_list.setMaximumHeight(110)
        self.sample_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {Theme.get("bg_log")};
                color: {Theme.get("text_secondary")};
                border: 1px solid {Theme.get("border")};
                border-radius: 4px;
                font-size: 11px;
            }}
        """)
        monitor_layout.addWidget(self.sample_list)

        # Loss 曲线
        self.loss_chart = LossChartWidget(title="")
        self.loss_chart.setMinimumHeight(200)
        self.loss_chart_label = QLabel("Loss 曲线")
        self.loss_chart_label.setStyleSheet(f"font-weight: bold; color: {Theme.get('text_primary')}; font-size: 12px;")
        monitor_layout.addWidget(self.loss_chart_label)
        monitor_layout.addWidget(self.loss_chart)

        # 日志
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

    def _browse_image_dir(self):
        """浏览图像数据集目录并统计图片数量"""
        path = QFileDialog.getExistingDirectory(self, "选择图像数据集目录")
        if not path:
            return
        self.image_dir_input.setText(path)
        self._update_image_count(path)

    def _update_image_count(self, dir_path):
        """统计目录中支持的图片数量"""
        import os
        extensions = (".jpg", ".jpeg", ".png", ".webp")
        count = 0
        for f in os.listdir(dir_path):
            if any(f.lower().endswith(ext) for ext in extensions):
                count += 1
        text = f"找到 {count} 张图片"
        if count == 0:
            text = "未找到图片"
            self.image_count_label.setStyleSheet(f"color: {Theme.get('red')}; font-size: 11px;")
        else:
            self.image_count_label.setStyleSheet(f"color: {Theme.get('green')}; font-size: 11px;")
        self.image_count_label.setText(text)
        # 切到图像目录时自动切换评估问题
        if count > 0:
            current_text = self.eval_questions_edit.toPlainText().strip()
            default_text = "\n".join(IMAGE_EVAL_QUESTIONS)
            if not current_text or current_text == "\n".join(EVAL_QUESTIONS):
                self.eval_questions_edit.setPlainText(default_text)
        return count

    def _init_eval_questions(self):
        """初始化评估问题：根据图像目录自动选择默认问题"""
        dir_path = self.image_dir_input.text().strip()
        if dir_path:
            import os
            if os.path.isdir(dir_path):
                self.eval_questions_edit.setPlainText("\n".join(IMAGE_EVAL_QUESTIONS))
                return
        self.eval_questions_edit.setPlainText("\n".join(EVAL_QUESTIONS))

    def refresh_models(self):
        """刷新模型下拉列表"""
        models = self.db.get_all_models()
        self.teacher_combo.clear()
        self.student_combo.clear()

        for m in models:
            text = f"{m['name']} ({m.get('parameters', 0):.3f}B)"
            self.teacher_combo.addItem(text, m["id"])
            self.student_combo.addItem(text, m["id"])

    def _read_config_values(self) -> dict:
        """读取全部可调参数（不含模型选择），供启动实验与模板复用"""
        return {
            "seed_questions": [q.strip() for q in self.seed_questions_edit.toPlainText().strip().split("\n") if q.strip()],
            "reference_answers": [r.strip() for r in self.reference_answers_edit.toPlainText().strip().split("\n") if r.strip()],
            "eval_questions": [q.strip() for q in self.eval_questions_edit.toPlainText().strip().split("\n") if q.strip()],
            "num_generate": self.num_generate_spin.value(),
            "temperature": self.temp_spin.value(),
            "top_p": self.top_p_spin.value(),
            "top_k": self.top_k_spin.value(),
            "repetition_penalty": self.rep_penalty_spin.value(),
            "num_beams": self.num_beams_spin.value(),
            "enable_think": self.enable_think_cb.isChecked(),
            "max_new_tokens": self.max_token_spin.value(),
            "image_dir": self.image_dir_input.text().strip(),
            "image_prompts": [p.strip() for p in self.image_prompts_edit.toPlainText().strip().split("\n") if p.strip()],
            "image_prompt_template": (self.image_prompts_edit.toPlainText().strip().split("\n") or [""])[0].strip(),
            "max_images": self.max_images_spin.value(),
            "use_4bit": self.use_4bit_cb.isChecked(),
            "quant_type": self.quant_type_combo.currentText(),
            "double_quant": self.double_quant_cb.isChecked(),
            "compute_dtype": self.compute_dtype_combo.currentText(),
            "lora_r": self.lora_r_spin.value(),
            "lora_alpha": self.lora_alpha_spin.value(),
            "lora_dropout": self.lora_dropout_spin.value(),
            "epochs": self.epochs_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "weight_decay": self.weight_decay_spin.value(),
            "warmup_steps": self.warmup_spin.value(),
            "grad_accum_steps": self.grad_accum_spin.value(),
            "optimizer": self.optimizer_combo.currentText().lower(),
            "batch_size": self.batch_spin.value(),
            "max_seq_len": self.max_seq_len_spin.value(),
            "grad_clip": self.grad_clip_spin.value(),
            "seed": self.seed_spin.value(),
            "eval_max_new_tokens": self.eval_max_tokens_spin.value(),
            "use_clip_score": self.use_clip_cb.isChecked(),
        }

    def _get_config(self):
        """获取完整实验配置（含模型选择）"""
        teacher_idx = self.teacher_combo.currentIndex()
        student_idx = self.student_combo.currentIndex()

        if teacher_idx < 0 or student_idx < 0:
            return None

        teacher_id = self.teacher_combo.itemData(teacher_idx)
        student_id = self.student_combo.itemData(student_idx)
        teacher_model = self.db.get_model_by_id(teacher_id)
        student_model = self.db.get_model_by_id(student_id)

        config = self._read_config_values()
        config.update({
            "teacher_model_id": teacher_id,
            "student_model_id": student_id,
            "teacher_path": teacher_model["local_path"] if teacher_model else "",
            "student_path": student_model["local_path"] if student_model else "",
        })
        return config

    # ---- 实验模板 ----

    def refresh_templates(self):
        """刷新模板下拉列表"""
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
        """把模板配置写回界面控件"""
        self.num_generate_spin.setValue(int(cfg.get("num_generate", 50)))
        self.temp_spin.setValue(float(cfg.get("temperature", 0.7)))
        self.top_p_spin.setValue(float(cfg.get("top_p", 0.9)))
        self.top_k_spin.setValue(int(cfg.get("top_k", 0)))
        self.rep_penalty_spin.setValue(float(cfg.get("repetition_penalty", 1.0)))
        self.num_beams_spin.setValue(int(cfg.get("num_beams", 1)))
        self.enable_think_cb.setChecked(bool(cfg.get("enable_think", True)))
        self.max_token_spin.setValue(int(cfg.get("max_new_tokens", 128)))
        self.image_dir_input.setText(cfg.get("image_dir", ""))
        prompts = cfg.get("image_prompts") or [cfg.get("image_prompt_template", "请详细描述这张图片的内容")]
        self.image_prompts_edit.setPlainText("\n".join(prompts))
        self.max_images_spin.setValue(int(cfg.get("max_images", 50)))
        self.use_4bit_cb.setChecked(bool(cfg.get("use_4bit", True)))
        self.quant_type_combo.setCurrentText(cfg.get("quant_type", "nf4"))
        self.double_quant_cb.setChecked(bool(cfg.get("double_quant", True)))
        self.compute_dtype_combo.setCurrentText(cfg.get("compute_dtype", "fp16"))
        self.lora_r_spin.setValue(int(cfg.get("lora_r", 16)))
        self.lora_alpha_spin.setValue(int(cfg.get("lora_alpha", 32)))
        self.lora_dropout_spin.setValue(float(cfg.get("lora_dropout", 0.05)))
        self.epochs_spin.setValue(int(cfg.get("epochs", 3)))
        self.lr_spin.setValue(float(cfg.get("learning_rate", 5e-5)))
        self.weight_decay_spin.setValue(float(cfg.get("weight_decay", 0.01)))
        self.warmup_spin.setValue(int(cfg.get("warmup_steps", 0)))
        self.grad_accum_spin.setValue(int(cfg.get("grad_accum_steps", 1)))
        self.max_seq_len_spin.setValue(int(cfg.get("max_seq_len", 256)))
        self.grad_clip_spin.setValue(float(cfg.get("grad_clip", 1.0)))
        opt = cfg.get("optimizer", "adamw")
        self.optimizer_combo.setCurrentText({"adamw": "AdamW", "adamw8bit": "AdamW8bit", "sgd": "SGD"}.get(opt, "AdamW"))
        self.seed_spin.setValue(int(cfg.get("seed", 42)))
        self.eval_max_tokens_spin.setValue(int(cfg.get("eval_max_new_tokens", 64)))
        self.use_clip_cb.setChecked(bool(cfg.get("use_clip_score", False)))
        if cfg.get("eval_questions"):
            self.eval_questions_edit.setPlainText("\n".join(cfg["eval_questions"]))
        if cfg.get("reference_answers"):
            self.reference_answers_edit.setPlainText("\n".join(cfg["reference_answers"]))
        if cfg.get("seed_questions"):
            self.seed_questions_edit.setPlainText("\n".join(cfg["seed_questions"]))
        if cfg.get("image_dir"):
            self._update_image_count(cfg["image_dir"])

    def _start_batch(self, *args):
        """批量交叉矩阵：模板 × 学生模型 顺序运行（教师固定）"""
        templates = list_templates()
        if not templates:
            QMessageBox.information(self, "提示", "请先保存至少一个实验模板，再批量运行")
            return
        models = self.db.get_all_models()
        if not models:
            QMessageBox.warning(self, "提示", "请先导入教师和学生模型")
            return
        picked = self._batch_picker(models, templates)
        if not picked or not picked["templates"] or not picked["students"]:
            return

        teacher_id = picked["teacher_id"]
        teacher_model = self.db.get_model_by_id(teacher_id)
        if not teacher_model:
            return
        configs = []
        for name in picked["templates"]:
            cfg = load_template(name)
            for student_id in picked["students"]:
                student_model = self.db.get_model_by_id(student_id)
                if not student_model:
                    continue
                exp_cfg = dict(cfg)
                exp_cfg.update({
                    "teacher_model_id": teacher_id,
                    "student_model_id": student_id,
                    "teacher_path": teacher_model["local_path"],
                    "student_path": student_model["local_path"],
                })
                exp_id = self.db.add_experiment(
                    name=f"批量[{name}] {teacher_model['name']} → {student_model['name']}",
                    exp_type="distillation",
                    config=exp_cfg,
                    teacher_model_id=teacher_id,
                    student_model_id=student_id,
                )
                exp_cfg["exp_id"] = exp_id
                exp_cfg["_batch_name"] = f"{name} → {student_model['name']}"
                configs.append(exp_cfg)

        if not configs:
            return

        self._save_session()

        self._batch = BatchWorker(configs)
        self._batch.batch_progress.connect(
            lambda cur, total, name: self.status_label.setText(f"批量 {cur}/{total}: {name}")
        )
        self._batch.batch_finished.connect(self._on_batch_finished)
        self._batch.log_updated.connect(self._on_log)
        self._batch.phase_updated.connect(self._on_phase)
        self._batch.sample_ready.connect(self._on_sample)
        self._batch.metric_updated.connect(self._on_metric)
        self._batch.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.loss_chart.clear_series()
        self.log_text.clear()
        self.sample_list.clear()
        self.progress_bar.setValue(0)
        self.phase_label.setText("阶段: 批量队列")
        self._on_log(
            f"批量实验启动: {len(configs)} 个组合"
            f"（模板 {len(picked['templates'])} × 学生 {len(picked['students'])}，教师 {teacher_model['name']}）"
        )

    def _batch_picker(self, models, templates):
        """弹出批量配置对话框：教师单选，学生/模板多选"""
        dialog = QDialog(self)
        dialog.setWindowTitle("批量交叉矩阵配置")
        dialog.setMinimumWidth(520)
        layout = QVBoxLayout(dialog)
        hint = QLabel("选择教师（单选）、学生模型与模板（多选），将按 模板×学生 顺序运行")
        hint.setStyleSheet(f"color: {Theme.get('text_muted')}; font-size: 11px;")
        layout.addWidget(hint)

        teacher_combo = QComboBox()
        current_teacher = self.teacher_combo.currentData()
        for m in models:
            teacher_combo.addItem(f"{m['name']} ({m.get('parameters', 0):.3f}B)", m["id"])
        if current_teacher:
            idx = teacher_combo.findData(current_teacher)
            if idx >= 0:
                teacher_combo.setCurrentIndex(idx)
        layout.addWidget(QLabel("教师模型:"))
        layout.addWidget(teacher_combo)

        student_list = QListWidget()
        student_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        for m in models:
            item = QListWidgetItem(f"{m['name']} ({m.get('parameters', 0):.3f}B)")
            item.setData(Qt.ItemDataRole.UserRole, m["id"])
            student_list.addItem(item)
        layout.addWidget(QLabel("学生模型（可多选）:"))
        layout.addWidget(student_list)

        template_list = QListWidget()
        template_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        for t in templates:
            item = QListWidgetItem(f"{t['name']}（{t['saved_at']}）")
            item.setData(Qt.ItemDataRole.UserRole, t["name"])
            template_list.addItem(item)
        layout.addWidget(QLabel("实验模板（可多选）:"))
        layout.addWidget(template_list)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("开始运行")
        cancel_btn = QPushButton("取消")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return {
            "teacher_id": teacher_combo.currentData(),
            "students": [
                student_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(student_list.count())
                if student_list.item(i).isSelected()
            ],
            "templates": [
                template_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(template_list.count())
                if template_list.item(i).isSelected()
            ],
        }

    def _on_batch_finished(self, ok_count, fail_count):
        self._on_log(f"批量完成: 成功 {ok_count}，失败 {fail_count}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"状态: 批量完成（成功 {ok_count} / 失败 {fail_count}）")

    def _start_experiment(self):
        config = self._get_config()
        if not config:
            QMessageBox.warning(self, "提示", "请先导入教师和学生模型")
            return

        self._save_session()

        # 创建实验记录
        teacher_model = self.db.get_model_by_id(config["teacher_model_id"])
        student_model = self.db.get_model_by_id(config["student_model_id"])
        exp_name = f"蒸馏 {teacher_model['name']} → {student_model['name']}"

        exp_id = self.db.add_experiment(
            name=exp_name,
            exp_type="distillation",
            config=config,
            teacher_model_id=config["teacher_model_id"],
            student_model_id=config["student_model_id"],
        )
        config["exp_id"] = exp_id
        self.current_exp_id = exp_id

        # 启动工作线程
        self.worker = ExperimentWorker("distillation", config)
        self.worker.status_updated.connect(self._on_status)
        self.worker.metric_updated.connect(self._on_metric)
        self.worker.log_updated.connect(self._on_log)
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.phase_updated.connect(self._on_phase)
        self.worker.sample_ready.connect(self._on_sample)
        self.worker.data_generated.connect(lambda n: self._on_log(f"数据生成完成: {n} 条"))
        self.worker.baseline_ready.connect(lambda d: self._on_log(f"基线评估完成"))
        self.worker.epoch_ended.connect(lambda e, m: self._on_log(f"Epoch {e} 完成, Loss: {m.get('loss', 0):.4f}"))
        self.worker.experiment_completed.connect(self._on_complete)
        self.worker.experiment_error.connect(self._on_error)

        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.loss_chart.clear_series()
        self.log_text.clear()
        self.sample_list.clear()
        self.progress_bar.setValue(0)
        self.phase_label.setText("阶段: -")
        self.speed_label.setText("速度: -")
        self.elapsed_label.setText("耗时: -")
        self.eta_label.setText("剩余: -")
        self._phase_total = 0
        self._on_log(f"实验 #{exp_id} 启动: {exp_name}")
        self.db.update_experiment_status(exp_id, "running")

    def _stop_experiment(self):
        if self._batch and self._batch.isRunning():
            self._batch.stop()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self._on_log("正在停止批量队列，当前实验完成后退出...")
            return
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self._on_log("正在停止实验，等待当前步骤完成后退出...")

    def stop(self):
        """应用关闭时调用：请求停止实验（协作式）"""
        if self._batch and self._batch.isRunning():
            self._batch.stop()
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def _on_status(self, msg):
        self.status_label.setText(f"状态: {msg}")

    def _on_metric(self, name, value, step):
        if name == "loss":
            self.loss_chart.update_data("loss", value, step)
            self._metrics.append({"epoch": step, "loss": value})
        elif name == "speed":
            self.speed_label.setText(f"速度: {value:.2f} it/s")
            total = getattr(self, "_phase_total", 0)
            remaining = max(0, total - step - 1) if total else 0
            if value > 0 and remaining > 0:
                eta_min = int(remaining / value / 60)
                eta_sec = int(remaining / value % 60)
                self.eta_label.setText(f"剩余: {eta_min}m{eta_sec:02d}s")

    def _on_phase(self, phase, current, total):
        self.phase_label.setText(f"阶段: {phase}")
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(min(current, total))
            self._phase_total = total
            pct = int(current / total * 100)
            self.progress_bar.setFormat(f"{phase} {current}/{total} ({pct}%)")

    def _on_sample(self, sample):
        """逐样本识别结果预览"""
        img = sample.get("image", "")
        q = sample.get("question", "")[:24]
        a = (sample.get("answer", "") or "").replace("\n", " ")[:80]
        prefix = f"[{img}] " if img else ""
        item = QListWidgetItem(f"{prefix}{q} → {a}")
        img_path = sample.get("image_path", "")
        if img_path and os.path.isfile(img_path):
            reader = QImageReader(img_path)
            reader.setScaledSize(QSize(44, 44))
            pixmap = QPixmap.fromImage(reader.read())
            if not pixmap.isNull():
                item.setIcon(QIcon(pixmap))
        self.sample_list.insertItem(0, item)
        while self.sample_list.count() > 200:
            self.sample_list.takeItem(self.sample_list.count() - 1)

    def _on_log(self, msg):
        self.log_text.append(msg)

    def _on_progress(self, current, total):
        self.status_label.setText(f"数据生成: {current}/{total}")

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
        self.loss_chart_label.setStyleSheet(f"font-weight: bold; color: {t['text_primary']}; font-size: 12px;")
        self.log_label.setStyleSheet(f"font-weight: bold; color: {t['text_primary']}; font-size: 12px;")
        self.image_count_label.setStyleSheet(f"color: {t['text_muted']}; font-size: 11px;")
        self.use_4bit_cb.setStyleSheet(f"color: {t['text_primary']}; font-size: 12px;")
        self.enable_think_cb.setStyleSheet(f"color: {t['text_primary']}; font-size: 12px;")
        self.double_quant_cb.setStyleSheet(f"color: {t['text_primary']}; font-size: 12px;")
        self.use_clip_cb.setStyleSheet(f"color: {t['text_primary']}; font-size: 12px;")
        self.phase_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 11px;")
        self.sample_label.setStyleSheet(f"color: {t['text_muted']}; font-size: 11px;")
        for lb in (self.speed_label, self.elapsed_label, self.eta_label):
            lb.setStyleSheet(f"color: {t['text_muted']}; font-size: 11px;")
        self.sample_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {t["bg_log"]};
                color: {t["text_secondary"]};
                border: 1px solid {t["border"]};
                border-radius: 4px;
                font-size: 11px;
            }}
        """)
        for edit in (self.eval_questions_edit, self.reference_answers_edit,
                     self.seed_questions_edit, self.image_prompts_edit):
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
                background-color: {t["green"]}; color: #ffffff;
                padding: 6px 16px; border-radius: 4px; font-weight: bold;
                font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {t["teal"]}; }}
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
        for section in self._sections:
            section.apply_theme()
        self.loss_chart.apply_theme()
