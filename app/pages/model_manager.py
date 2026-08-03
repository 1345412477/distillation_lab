"""模型管理页面 - 模型列表、导入、详情"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QTableWidget, QTableWidgetItem,
                              QHeaderView, QMessageBox, QAbstractItemView)
from PyQt6.QtCore import Qt
from ..models.db_manager import DBManager
from ..components.import_dialog import ImportModelDialog
from ..utils.theme import Theme


class ModelManagerPage(QWidget):
    """模型管理页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.db = DBManager()
        self._last_import = {"source_idx": 0, "repo_id": "", "name": "", "local_path": ""}
        self.init_ui()
        self.refresh()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # 顶部标题和操作按钮
        header_layout = QHBoxLayout()
        self.title = QLabel("模型管理")
        self.title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {Theme.get('text_primary')};")
        header_layout.addWidget(self.title)

        header_layout.addStretch()

        self.import_btn = QPushButton("+ 导入模型")
        self.import_btn.clicked.connect(self._import_model)
        self.import_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.get("accent")}; color: #ffffff;
                padding: 8px 18px; border-radius: 4px; font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {Theme.get("accent_hover")}; }}
        """)
        header_layout.addWidget(self.import_btn)

        self.delete_btn = QPushButton("删除模型")
        self.delete_btn.clicked.connect(self._delete_model)
        self.delete_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.get("red")}; color: #ffffff;
                padding: 8px 14px; border-radius: 4px; font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {Theme.get("red")}; }}
        """)
        header_layout.addWidget(self.delete_btn)

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh)
        self.refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.get("btn_default")}; color: {Theme.get("btn_default_text")};
                padding: 8px 14px; border-radius: 4px;
            }}
            QPushButton:hover {{ background-color: {Theme.get("btn_default_hover")}; }}
        """)
        header_layout.addWidget(self.refresh_btn)

        layout.addLayout(header_layout)

        # 模型表格
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "ID", "模型名称", "来源", "参数量", "架构", "文件大小", "导入时间"
        ])
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {Theme.get("bg_secondary")}; color: {Theme.get("text_primary")};
                gridline-color: {Theme.get("border")}; border: 1px solid {Theme.get("border")};
                border-radius: 6px;
            }}
            QTableWidget::item {{ padding: 6px; }}
            QHeaderView::section {{
                background-color: {Theme.get("header_bg")}; color: {Theme.get("text_primary")};
                padding: 8px; border: 1px solid {Theme.get("border")};
                font-weight: bold;
            }}
            QTableWidget::item:alternate {{ background-color: {Theme.get("table_alt")}; }}
            QTableWidget::item:selected {{ background-color: {Theme.get("bg_surface")}; }}
        """)
        self.table.cellDoubleClicked.connect(self._show_model_detail)
        layout.addWidget(self.table)

    def refresh(self):
        """刷新模型列表"""
        models = self.db.get_all_models()
        self.table.setRowCount(len(models))

        for row, model in enumerate(models):
            self.table.setItem(row, 0, QTableWidgetItem(str(model["id"])))
            self.table.setItem(row, 1, QTableWidgetItem(model.get("name", "")))
            source_map = {"modelscope": "ModelScope", "huggingface": "HF Hub", "local": "本地"}
            self.table.setItem(row, 2, QTableWidgetItem(source_map.get(model.get("source", ""), model.get("source", ""))))
            params = model.get("parameters", 0)
            self.table.setItem(row, 3, QTableWidgetItem(f"{params:.3f}B" if params else "-"))
            self.table.setItem(row, 4, QTableWidgetItem(model.get("architecture", "-")))
            self.table.setItem(row, 5, QTableWidgetItem(model.get("file_size", "-")))
            self.table.setItem(row, 6, QTableWidgetItem(model.get("created_at", "-")[:16]))

    def _import_model(self):
        """打开导入对话框"""
        dialog = ImportModelDialog(self, last_import=self._last_import)
        if dialog.exec():
            # 保存本次导入的配置，方便下次快速重新导入
            self._last_import = dialog.get_import_config()
            self.refresh()

    def _delete_model(self):
        """删除选中模型"""
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "提示", "请先选中要删除的模型")
            return

        model_id_item = self.table.item(row, 0)
        name_item = self.table.item(row, 1)
        if not model_id_item:
            return

        model_id = int(model_id_item.text())
        model_name = name_item.text() if name_item else f"#{model_id}"

        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除模型 [{model_name}] 吗？\n此操作不可恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.db.delete_model(model_id)
        QMessageBox.information(self, "成功", f"模型 [{model_name}] 已删除")
        self.refresh()

    def _show_model_detail(self, row, col):
        """显示模型详情"""
        model_id_item = self.table.item(row, 0)
        if not model_id_item:
            return
        model_id = int(model_id_item.text())
        model = self.db.get_model_by_id(model_id)
        if not model:
            return

        detail_text = (
            f"模型名称: {model.get('name', '')}\n"
            f"来源: {model.get('source', '')}\n"
            f"Repo ID: {model.get('repo_id', '')}\n"
            f"本地路径: {model.get('local_path', '')}\n"
            f"架构: {model.get('architecture', '')}\n"
            f"参数量: {model.get('parameters', 0):.3f}B\n"
            f"层数: {model.get('num_layers', 0)}\n"
            f"隐藏维度: {model.get('hidden_size', 0)}\n"
            f"精度: {model.get('dtype', '')}\n"
            f"文件大小: {model.get('file_size', '')}\n"
            f"导入时间: {model.get('created_at', '')}"
        )

        QMessageBox.information(self, f"模型详情 - {model.get('name', '')}", detail_text)

    def apply_theme(self):
        """应用当前主题"""
        self.setStyleSheet("")
        self.title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {Theme.get('text_primary')};")
        self.import_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.get("accent")}; color: #ffffff;
                padding: 8px 18px; border-radius: 4px; font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {Theme.get("accent_hover")}; }}
        """)
        self.delete_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.get("red")}; color: #ffffff;
                padding: 8px 14px; border-radius: 4px; font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {Theme.get("red")}; }}
        """)
        self.refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Theme.get("btn_default")}; color: {Theme.get("btn_default_text")};
                padding: 8px 14px; border-radius: 4px;
            }}
            QPushButton:hover {{ background-color: {Theme.get("btn_default_hover")}; }}
        """)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {Theme.get("bg_secondary")}; color: {Theme.get("text_primary")};
                gridline-color: {Theme.get("border")}; border: 1px solid {Theme.get("border")};
                border-radius: 6px;
            }}
            QTableWidget::item {{ padding: 6px; }}
            QHeaderView::section {{
                background-color: {Theme.get("header_bg")}; color: {Theme.get("text_primary")};
                padding: 8px; border: 1px solid {Theme.get("border")};
                font-weight: bold;
            }}
            QTableWidget::item:alternate {{ background-color: {Theme.get("table_alt")}; }}
            QTableWidget::item:selected {{ background-color: {Theme.get("bg_surface")}; }}
        """)
