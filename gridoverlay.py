from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor

class GridOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.parent = parent
        self.zoomed_grid = None
        self.zoomed_number = None
        self.grid_visible = False

    def show_grid(self):
        if self.parent:
            self.setGeometry(self.parent.geometry())
            self.grid_visible = True
            self.zoomed_grid = None
            self.zoomed_number = None
            self.show()
            self.update()

    def zoom_grid(self, number):
        if 1 <= number <= 9:
            self.zoomed_number = number
            self.zoomed_grid = self.calculate_grid_rect(number)
            self.update()

    def calculate_grid_rect(self, number):
        width = self.width() // 3
        height = self.height() // 3
        row = (number - 1) // 3
        col = (number - 1) % 3
        return QRect(col * width, row * height, width, height)

    def paintEvent(self, event):
        if not self.grid_visible:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.zoomed_grid:
            self.draw_zoomed_grid(painter)
        else:
            self.draw_main_grid(painter)

    def draw_main_grid(self, painter):
        width = self.width() // 3
        height = self.height() // 3

        for i in range(9):
            row = i // 3
            col = i % 3
            rect = QRect(col * width, row * height, width, height)
            self.draw_grid_cell(painter, rect, i + 1)

    def draw_zoomed_grid(self, painter):
        rect = self.zoomed_grid
        width = rect.width() // 3
        height = rect.height() // 3

        for i in range(9):
            row = i // 3
            col = i % 3
            sub_rect = QRect(rect.x() + col * width, rect.y() + row * height, width, height)
            self.draw_grid_cell(painter, sub_rect, i + 1)

    def draw_grid_cell(self, painter, rect, number):
        painter.setPen(QPen(QColor(255, 255, 255, 200), 2))
        painter.setBrush(QBrush(QColor(0, 0, 0, 100)))
        painter.drawRect(rect)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(rect, Qt.AlignCenter, str(number))

    def close_grid(self):
        self.grid_visible = False
        self.hide()
