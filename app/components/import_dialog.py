"""导入模型对话框 - 支持三种导入方式"""
import os
from PyQt6.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QLineEdit, QComboBox, QPushButton, QFileDialog,
                              QGroupBox, QFormLayout, QMessageBox)
from PyQt6.QtCore import Qt
from ..models.db_manager import DBManager
from ..models.model_manager import parse_model_config, get_model_path, resolve_model_path
from ..utils.config import MODEL_SOURCE_CHOICES
from ..utils.theme import Theme


class ImportModelDialog(QDialog):
    """模型导入对话框"""

    def __init__(self, parent=None, last_import=None):
        super().__init__(parent)
        self.setWindowTitle("导入模型")
        self.setMinimumWidth(500)
        self.db = DBManager()
        self._result = None
        self._last_import = last_import or {}
        self.init_ui()
        self._restore_last_import()

    def apply_theme(self):
        """应用当前主题"""
        self.import_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.get("accent")}; color: {Theme.get("bg_primary")};
                padding: 8px 20px; border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {Theme.get("accent_hover")}; }}
        """)
        cancel_btn = self.findChild(QPushButton, "")
        # 找到取消按钮并更新样式
        for btn in self.findChildren(QPushButton):
            if btn.text() == "取消":
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {Theme.get("btn_default")}; color: {Theme.get("btn_default_text")};
                        padding: 8px 20px; border-radius: 4px;
                    }}
                    QPushButton:hover {{ background-color: {Theme.get("btn_default_hover")}; }}
                """)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 导入方式选择
        source_group = QGroupBox("模型来源")
        source_layout = QVBoxLayout(source_group)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["ModelScope（国内加速）", "HuggingFace Hub", "本地路径"])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        source_layout.addWidget(self.source_combo)
        layout.addWidget(source_group)

        # 模型标识
        id_group = QGroupBox("模型标识")
        id_layout = QFormLayout(id_group)

        self.repo_id_input = QLineEdit()
        self.repo_id_input.setPlaceholderText("例如: Qwen/Qwen3-1.7B")
        id_layout.addRow("Repo ID:", self.repo_id_input)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("留空则使用 Repo ID")
        id_layout.addRow("显示名称:", self.name_input)

        # 本地路径选择（默认隐藏）
        self.path_layout_widget = QWidget()
        path_layout = QHBoxLayout(self.path_layout_widget)
        path_layout.setContentsMargins(0, 0, 0, 0)
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("选择本地模型目录...")
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_path)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(browse_btn)
        id_layout.addRow("本地路径:", self.path_layout_widget)
        self.path_layout_widget.hide()

        layout.addWidget(id_group)

        # 按钮
        btn_layout = QHBoxLayout()
        self.import_btn = QPushButton("导入并注册")
        self.import_btn.clicked.connect(self._do_import)
        self.import_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.get("accent")}; color: {Theme.get("bg_primary")};
                padding: 8px 20px; border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {Theme.get("accent_hover")}; }}
        """)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.get("btn_default")}; color: {Theme.get("btn_default_text")};
                padding: 8px 20px; border-radius: 4px;
            }}
            QPushButton:hover {{ background-color: {Theme.get("btn_default_hover")}; }}
        """)

        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(self.import_btn)
        layout.addLayout(btn_layout)

    def _on_source_changed(self, index):
        """切换导入方式"""
        if index == 2:  # 本地路径
            self.path_layout_widget.show()
            self.repo_id_input.setPlaceholderText("模型目录名称（可选）")
        else:
            self.path_layout_widget.hide()
            self.repo_id_input.setPlaceholderText("例如: Qwen/Qwen3-1.7B")

    def _browse_path(self):
        """浏览本地路径"""
        path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if path:
            self.path_input.setText(path)

    def _restore_last_import(self):
        """恢复上次导入的配置"""
        if not self._last_import:
            return
        # 恢复来源选择
        source_idx = self._last_import.get("source_idx", 0)
        if 0 <= source_idx < self.source_combo.count():
            self.source_combo.setCurrentIndex(source_idx)
        # 恢复 repo_id
        repo_id = self._last_import.get("repo_id", "")
        if repo_id:
            self.repo_id_input.setText(repo_id)
        # 恢复名称
        name = self._last_import.get("name", "")
        if name:
            self.name_input.setText(name)
        # 恢复本地路径
        local_path = self._last_import.get("local_path", "")
        if local_path:
            self.path_input.setText(local_path)

    def get_import_config(self):
        """获取当前导入配置，用于下次快速重新导入"""
        return {
            "source_idx": self.source_combo.currentIndex(),
            "repo_id": self.repo_id_input.text().strip(),
            "name": self.name_input.text().strip(),
            "local_path": self.path_input.text().strip(),
        }

    def _do_import(self):
        """执行导入"""
        source_idx = self.source_combo.currentIndex()
        sources = {0: "modelscope", 1: "huggingface", 2: "local"}
        source = sources[source_idx]

        repo_id = self.repo_id_input.text().strip()
        local_path = self.path_input.text().strip() if source == "local" else ""
        name = self.name_input.text().strip() or repo_id

        # 严格校验：非本地模式必须填写 Repo ID
        if source != "local":
            if not repo_id:
                QMessageBox.warning(self, "提示", "请输入 Repo ID（例如 Qwen/Qwen3-1.7B）")
                return
            if "/" not in repo_id:
                QMessageBox.warning(self, "提示",
                    f"Repo ID 格式不正确\n"
                    f"正确格式：组织名/模型名（例如 Qwen/Qwen3-1.7B）\n"
                    f"你输入的是：{repo_id}")
                return
        elif source == "local" and not local_path:
            QMessageBox.warning(self, "提示", "请选择本地模型路径")
            return

        try:
            # 下载模型
            if source != "local":
                self.import_btn.setEnabled(False)
                self.import_btn.setText(f"正在下载 {repo_id}...")
                self.repaint()
                local_path = get_model_path(repo_id, source)
                if not local_path or not os.path.exists(local_path):
                    raise FileNotFoundError(f"模型下载失败，路径不存在: {local_path}")
                # 将 ModelScope 缓存根目录解析为实际 snapshot 子目录
                local_path = resolve_model_path(local_path)

            # 解析模型配置
            model_info = parse_model_config(local_path)

            # 名称为空时，自动使用架构名作为默认名称
            if not name:
                arch = model_info.get("architecture", "unknown")
                name = arch.split("For")[0] if "For" in arch else arch
                if not name:
                    name = os.path.basename(local_path)

            # 检查解析结果是否有效
            if model_info.get("parameters", 0) == 0:
                QMessageBox.warning(self, "警告",
                    f"模型配置解析异常！\n"
                    f"路径: {local_path}\n"
                    f"可能原因：config.json 文件缺失或格式不正确\n"
                    f"是否继续注册？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                # 如果用户选择继续，仍会注册但参数为0

            # 注册到数据库
            model_id = self.db.add_model(
                name=name,
                source=source,
                repo_id=repo_id,
                local_path=local_path,
                architecture=model_info.get("architecture", ""),
                parameters=model_info.get("parameters", 0),
                num_layers=model_info.get("num_layers", 0),
                hidden_size=model_info.get("hidden_size", 0),
                dtype=model_info.get("dtype", ""),
                file_size=model_info.get("file_size", ""),
            )

            self._result = {
                "id": model_id,
                "name": name,
                "source": source,
                "repo_id": repo_id,
                "local_path": local_path,
                "parameters": model_info.get("parameters", 0),
            }
            QMessageBox.information(self, "成功",
                f"模型 [{name}] 导入成功！\n"
                f"参数量: {model_info.get('parameters', 0):.3f}B\n"
                f"架构: {model_info.get('architecture', 'unknown')}")
            self.accept()

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            QMessageBox.critical(self, "导入失败", 
                f"模型导入出错:\n{str(e)}\n\n详细信息:\n{error_detail}")
        finally:
            self.import_btn.setEnabled(True)
            self.import_btn.setText("导入并注册")

    def get_result(self):
        """获取导入结果"""
        return self._result
