"""主题系统 - 深色/浅色模式颜色定义"""
from PyQt6.QtWidgets import QApplication


class Theme:
    """全局主题管理，支持深色/浅色模式切换"""

    _current = "dark"

    _colors = {
        "dark": {
            "bg_primary": "#0a0a0a",
            "bg_secondary": "#141414",
            "bg_tertiary": "#1e1e1e",
            "bg_surface": "#2a2a2a",
            "bg_hover": "#333333",
            "bg_card": "#1a1a1a",
            "bg_log": "#111111",
            "text_primary": "#f0f0f0",
            "text_secondary": "#d0d0d0",
            "text_muted": "#909090",
            "border": "#333333",
            "border_light": "#404040",
            "accent": "#89b4fa",
            "accent_hover": "#b4befe",
            "green": "#a6e3a1",
            "yellow": "#f9e2af",
            "red": "#f38ba8",
            "purple": "#cba6f7",
            "teal": "#94e2d5",
            "peach": "#fab387",
            "header_bg": "#1a1a1a",
            "table_alt": "#1a1a1a",
            "btn_default": "#2a2a2a",
            "btn_default_hover": "#3a3a3a",
            "btn_default_text": "#f0f0f0",
            "scrollbar_bg": "#1a1a1a",
            "scrollbar_fg": "#404040",
        },
        "light": {
            "bg_primary": "#fafbfc",
            "bg_secondary": "#f0f2f5",
            "bg_tertiary": "#e8ebf0",
            "bg_surface": "#dfe3e8",
            "bg_hover": "#e8ebf0",
            "bg_card": "#ffffff",
            "bg_log": "#f5f7fa",
            "text_primary": "#1f2937",
            "text_secondary": "#4b5563",
            "text_muted": "#6b7280",
            "border": "#d1d5db",
            "border_light": "#e5e7eb",
            "accent": "#3b82f6",
            "accent_hover": "#2563eb",
            "green": "#10b981",
            "yellow": "#f59e0b",
            "red": "#ef4444",
            "purple": "#8b5cf6",
            "teal": "#14b8a6",
            "peach": "#f97316",
            "header_bg": "#f0f2f5",
            "table_alt": "#f9fafb",
            "btn_default": "#e5e7eb",
            "btn_default_hover": "#d1d5db",
            "btn_default_text": "#1f2937",
            "scrollbar_bg": "#e5e7eb",
            "scrollbar_fg": "#9ca3af",
        },
    }

    @classmethod
    def get(cls, key: str) -> str:
        """获取当前主题的配色值"""
        return cls._colors[cls._current].get(key, "#000000")

    @classmethod
    def toggle(cls) -> str:
        """切换主题，返回新模式名称"""
        cls._current = "light" if cls._current == "dark" else "dark"
        return cls._current

    @classmethod
    def mode(cls) -> str:
        return cls._current

    @classmethod
    def set_mode(cls, mode: str):
        cls._current = mode

    @classmethod
    def global_style(cls) -> str:
        """生成全局 QSS 样式表"""
        t = cls._colors[cls._current]
        return f"""
            QMainWindow {{ background-color: {t["bg_primary"]}; }}
            QWidget {{ color: {t["text_primary"]}; font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif; }}
            QStatusBar {{
                background-color: {t["header_bg"]}; color: {t["text_muted"]};
                border-top: 1px solid {t["border"]}; font-size: 11px;
            }}
            QTableWidget {{
                background-color: {t["bg_secondary"]}; color: {t["text_primary"]};
                gridline-color: {t["border"]}; border: 1px solid {t["border"]};
                border-radius: 6px; alternate-background-color: {t["table_alt"]};
            }}
            QTableWidget::item {{ padding: 6px; }}
            QTableWidget::item:selected {{ background-color: {t["bg_surface"]}; }}
            QHeaderView::section {{
                background-color: {t["header_bg"]}; color: {t["text_secondary"]};
                padding: 8px; border: 1px solid {t["border"]};
                font-weight: bold;
            }}
            QTextEdit {{
                background-color: {t["bg_log"]}; color: {t["text_secondary"]};
                border: 1px solid {t["border"]}; border-radius: 4px;
                font-family: 'Consolas', 'Courier New', monospace; font-size: 11px;
            }}
            QGroupBox {{
                font-weight: bold; color: {t["text_primary"]};
                border: 1px solid {t["border"]}; border-radius: 6px;
                margin-top: 8px; padding: 12px 8px 8px 8px;
                background-color: {t["bg_tertiary"]};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; subcontrol-position: top left;
                padding: 2px 8px; color: {t["text_primary"]};
            }}
            QComboBox {{
                background-color: {t["bg_secondary"]}; color: {t["text_primary"]};
                border: 1px solid {t["border"]}; border-radius: 4px;
                padding: 4px 8px; min-height: 24px;
            }}
            QComboBox:hover {{ border-color: {t["accent"]}; }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background-color: {t["bg_secondary"]}; color: {t["text_primary"]};
                selection-background-color: {t["bg_surface"]};
            }}
            QSpinBox, QDoubleSpinBox {{
                background-color: {t["bg_secondary"]}; color: {t["text_primary"]};
                border: 1px solid {t["border"]}; border-radius: 4px;
                padding: 4px; min-height: 24px;
            }}
            QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {t["accent"]}; }}
            QScrollArea {{ border: none; }}
            QSplitter::handle {{ background-color: {t["border"]}; }}
            QProgressBar {{
                background-color: {t["bg_surface"]}; border: none; border-radius: 3px;
                height: 18px; text-align: center; color: {t["text_primary"]}; font-size: 10px;
            }}
            QProgressBar::chunk {{
                background-color: {t["accent"]}; border-radius: 3px;
            }}
            QScrollBar:vertical {{
                background: {t["scrollbar_bg"]}; width: 10px; border: none; border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{
                background: {t["scrollbar_fg"]}; border-radius: 5px; min-height: 24px;
            }}
            QScrollBar::handle:vertical:hover {{ background: {t["accent"]}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar:horizontal {{
                background: {t["scrollbar_bg"]}; height: 10px; border: none; border-radius: 5px;
            }}
            QScrollBar::handle:horizontal {{
                background: {t["scrollbar_fg"]}; border-radius: 5px; min-width: 24px;
            }}
            QScrollBar::handle:horizontal:hover {{ background: {t["accent"]}; }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
            QToolTip {{
                background-color: {t["bg_surface"]}; color: {t["text_primary"]};
                border: 1px solid {t["border"]}; padding: 4px;
            }}
        """

    @classmethod
    def apply(cls):
        """将主题应用到 QApplication"""
        app = QApplication.instance()
        if app:
            app.setStyleSheet(cls.global_style())
