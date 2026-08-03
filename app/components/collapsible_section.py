"""可折叠分组组件 - 高级参数默认收起，保持界面简洁"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QToolButton, QFrame)
from PyQt6.QtCore import Qt
from ..utils.theme import Theme


class CollapsibleSection(QWidget):
    """标题按钮 + 内容区，点击展开/收起"""

    def __init__(self, title="", collapsed=True, parent=None):
        super().__init__(parent)
        self._expanded = not collapsed

        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)

        self._btn = QToolButton()
        self._btn.setText(title)
        self._btn.setCheckable(True)
        self._btn.setChecked(self._expanded)
        self._btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._btn.setArrowType(
            Qt.ArrowType.DownArrow if self._expanded else Qt.ArrowType.RightArrow
        )
        self._btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn.clicked.connect(self._toggle)
        layout.addWidget(self._btn)

        self._content = QFrame()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(10, 2, 0, 0)
        self._content_layout.setSpacing(2)
        self._content.setVisible(self._expanded)
        layout.addWidget(self._content)

        self.apply_theme()

    def _toggle(self, checked):
        self._expanded = checked
        self._btn.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self._content.setVisible(checked)

    def add_widget(self, widget):
        self._content_layout.addWidget(widget)

    def add_layout(self, layout):
        self._content_layout.addLayout(layout)

    def apply_theme(self):
        t = Theme._colors[Theme.mode()]
        self._btn.setStyleSheet(f"""
            QToolButton {{
                color: {t["text_secondary"]};
                background-color: transparent;
                border: none;
                padding: 2px 4px;
                font-size: 11px;
                font-weight: bold;
            }}
            QToolButton:hover {{ color: {t["accent"]}; }}
        """)
