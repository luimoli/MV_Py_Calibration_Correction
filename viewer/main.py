import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout, QLabel


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        grid = QGridLayout()
        grid.setSpacing(10)

        x = 0
        y = 0

        self.text1 = "x: {0},  y: {1}".format(x, y)
        self.label1 = QLabel(self.text1, self)
        grid.addWidget(self.label1, 0, 0, Qt.AlignTop)

        self.text2 = "x: {0},  y: {1}".format(x, y)

        self.label2 = QLabel(self.text2, self)
        grid.addWidget(self.label2, 0, 0, Qt.AlignTop)

        self.setMouseTracking(True)

        self.setLayout(grid)

        self.setGeometry(300, 300, 350, 200)
        self.setWindowTitle('Event object')
        self.show()

    def mouseMoveEvent(self, event2):
        x = event2.x()
        y = event2.y()

        text = "x: {0},  y: {1}".format(x, y)
        self.label2.setText(text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())