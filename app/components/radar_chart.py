"""雷达图 - 使用 pyqtgraph 极坐标模式绘制能力对比雷达图"""
import pyqtgraph as pg
import math
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from ..utils.theme import Theme


class RadarChartWidget(QWidget):
    """能力雷达图组件 - 多维度能力对比"""

    def __init__(self, title="能力雷达图", parent=None):
        super().__init__(parent)
        self._title = title
        self._dimensions = []
        self._data_series = {}
        self._colors = ["#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8", "#cba6f7"]

        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.header = QLabel(self._title)
        self.header.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {Theme.get('text_primary')}; padding: 4px;")
        self.header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 使用 PlotWidget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(Theme.get("bg_secondary"))
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.hideAxis("left")
        self.plot_widget.setRange(xRange=[-1.3, 1.3], yRange=[-1.3, 1.3])

        layout.addWidget(self.header)
        layout.addWidget(self.plot_widget)

        # 图例区域
        self.legend_label = QLabel()
        self.legend_label.setStyleSheet(f"color: {Theme.get('text_primary')}; font-size: 11px; padding: 2px;")
        self.legend_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.legend_label)

    def set_data(self, dimensions: list, series: dict):
        """
        设置雷达图数据
        dimensions: ["维度1", "维度2", ...]
        series: {"系列名": [0.8, 0.6, ...], ...}
        """
        self._dimensions = dimensions
        self._data_series = series
        self._render()

    def apply_theme(self):
        """应用当前主题"""
        bg = Theme.get("bg_secondary")
        text_primary = Theme.get("text_primary")

        # pyqtgraph PlotWidget 背景：使用 QColor 对象确保正确渲染
        self.plot_widget.setBackground(QColor(bg))
        vb = self.plot_widget.getViewBox()
        vb.setBackgroundColor(QColor(bg))
        vb.update()

        self.header.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {text_primary}; padding: 4px;")
        self.legend_label.setStyleSheet(f"color: {text_primary}; font-size: 11px; padding: 2px;")
        if self._dimensions:
            self._render()

    def _render(self):
        self.plot_widget.clear()

        if not self._dimensions or not self._data_series:
            return

        n = len(self._dimensions)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()

        # 绘制背景网格
        for level in [0.2, 0.4, 0.6, 0.8, 1.0]:
            pts = [(level * math.cos(a), level * math.sin(a)) for a in angles]
            pts.append(pts[0])  # 闭合
            grid = pg.PlotDataItem(
                [p[0] for p in pts], [p[1] for p in pts],
                pen=pg.mkPen(color=Theme.get("border"), width=1),
                fillLevel=None,
            )
            self.plot_widget.addItem(grid)

            # 标度文字
            if level > 0:
                text = pg.TextItem(f"{level:.0%}", color=Theme.get("text_muted"),
                                   anchor=(0.5, 0.5))
                text.setPos(level * math.cos(angles[0]), level * math.sin(angles[0]))
                self.plot_widget.addItem(text)

        # 绘制维度标签
        for i, dim in enumerate(self._dimensions):
            a = angles[i]
            label = pg.TextItem(dim, color=Theme.get("text_primary"), anchor=(0.5, 0.5))
            label.setPos(1.15 * math.cos(a), 1.15 * math.sin(a))
            self.plot_widget.addItem(label)

        # 绘制轴线
        for a in angles:
            line = pg.PlotDataItem(
                [0, math.cos(a)], [0, math.sin(a)],
                pen=pg.mkPen(color=Theme.get("border"), width=1),
            )
            self.plot_widget.addItem(line)

        # 绘制数据
        color_idx = 0
        legend_items = []
        for name, values in self._data_series.items():
            if len(values) != n:
                continue

            # 归一化到 [0, 1]
            max_val = max(values) if max(values) > 0 else 1.0
            norm_values = [v / max_val for v in values]

            pts = [(norm_values[i] * math.cos(angles[i]),
                    norm_values[i] * math.sin(angles[i])) for i in range(n)]
            pts.append(pts[0])

            color = self._colors[color_idx % len(self._colors)]
            color_idx += 1

            data_item = pg.PlotDataItem(
                [p[0] for p in pts], [p[1] for p in pts],
                pen=pg.mkPen(color=color, width=2),
                fillLevel=0,
                brush=pg.mkBrush(color + "60"),
                fillOutline=True,
            )
            self.plot_widget.addItem(data_item)

            legend_items.append(f'<span style="color:{color}">■</span> {name}')

        # 更新图例
        self.legend_label.setText("  " .join(legend_items))

    def add_series(self, name: str, values: list, dimensions: list = None):
        """添加或更新数据系列"""
        if dimensions:
            self._dimensions = dimensions
        self._data_series[name] = values
        self._render()

    def clear(self):
        """清除所有数据"""
        self._dimensions = []
        self._data_series = {}
        self._render()
