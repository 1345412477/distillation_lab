"""Loss 曲线图 - 基于 pyqtgraph 的实时折线图"""
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from ..utils.theme import Theme


class LossChartWidget(pg.PlotWidget):
    """实时 Loss 曲线图"""

    def __init__(self, title="Loss 曲线", parent=None):
        super().__init__(parent)
        self._title_text = title
        self.setBackground(Theme.get("bg_secondary"))
        self.setLabel("left", "Loss", color=Theme.get("text_secondary"))
        self.setLabel("bottom", "Step", color=Theme.get("text_secondary"))
        self.setTitle(title, color=Theme.get("text_primary"))
        self.showGrid(x=True, y=True, alpha=0.3)

        # 图例
        self.addLegend(offset=(10, 10))

        # 数据存储
        self._curves = {}
        self._x_data = {}
        self._y_data = {}

        # Y 轴范围自适应
        self._auto_y = True
        self._min_y = float("inf")
        self._max_y = float("-inf")

    def add_series(self, name: str, color=None):
        """添加一条数据线"""
        if color is None:
            color = self._get_color(len(self._curves))
        pen = pg.mkPen(color=color, width=2)
        curve = self.plot(pen=pen, name=name)
        self._curves[name] = curve
        self._x_data[name] = []
        self._y_data[name] = []

    def update_data(self, name: str, y_value: float, x_value: int = None):
        """更新数据点"""
        if name not in self._curves:
            self.add_series(name)

        if x_value is None:
            x_value = len(self._y_data[name])

        self._x_data[name].append(x_value)
        self._y_data[name].append(y_value)
        self._curves[name].setData(self._x_data[name], self._y_data[name])

        # 更新 Y 轴范围
        if y_value < self._min_y:
            self._min_y = y_value
        if y_value > self._max_y:
            self._max_y = y_value
        if self._auto_y and self._min_y != float("inf"):
            margin = (self._max_y - self._min_y) * 0.2 or 0.5
            self.setYRange(max(0, self._min_y - margin), self._max_y + margin)

    def apply_theme(self):
        """应用当前主题"""
        bg = Theme.get("bg_secondary")
        text = Theme.get("text_secondary")
        text_primary = Theme.get("text_primary")
        border = Theme.get("border")
        self.setBackground(bg)
        self.setLabel("left", "Loss", color=text)
        self.setLabel("bottom", "Step", color=text)
        # 更新标题颜色 - 直接修改已有 TitleItem 的样式
        try:
            if self.titleLabel:
                self.titleLabel.setStyle(color=text_primary)
        except AttributeError:
            pass
        self.getAxis("left").setPen(pg.mkPen(color=text))
        self.getAxis("bottom").setPen(pg.mkPen(color=text))
        self.getAxis("left").setTextPen(pg.mkPen(color=text))
        self.getAxis("bottom").setTextPen(pg.mkPen(color=text))
        # 浅色模式下网格线更明显
        alpha = 0.5 if Theme.mode() == "light" else 0.3
        self.showGrid(x=True, y=True, alpha=alpha)

    def clear_series(self):
        """清除所有数据"""
        for name in list(self._curves.keys()):
            self.removeItem(self._curves[name])
        self._curves.clear()
        self._x_data.clear()
        self._y_data.clear()
        self._min_y = float("inf")
        self._max_y = float("-inf")

    def remove_series(self, name: str):
        """移除指定数据线"""
        if name in self._curves:
            self.removeItem(self._curves[name])
            del self._curves[name]
            del self._x_data[name]
            del self._y_data[name]

    def _get_color(self, index):
        colors = [
            "#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8",
            "#cba6f7", "#94e2d5", "#fab387", "#b4befe",
        ]
        return colors[index % len(colors)]


class LossChartWidgetWithContainer(QWidget):
    """带容器的 Loss 图表"""

    def __init__(self, title="Loss 曲线"):
        super().__init__()
        self._title = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.header = QLabel(title)
        self.header.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {Theme.get('text_primary')}; padding: 4px;")
        self.header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.chart = LossChartWidget(title="")

        layout.addWidget(self.header)
        layout.addWidget(self.chart)

        # 初始化时应用主题
        self.apply_theme()

    def apply_theme(self):
        """应用当前主题"""
        self.header.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {Theme.get('text_primary')}; padding: 4px;")
        self.chart.apply_theme()
