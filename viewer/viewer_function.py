import cv2
import threading
from PyQt5.QtCore import QFile
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5 import QtGui
import PyQt5
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class ViewerFunc:
    def __init__(self, ui, mainWnd):

        self.ui = ui
        self.mainWnd = mainWnd
        self.ui.label.mouseMoveEvent = self.mouseMoveEvent

        self.ui.centralwidget.dragEnterEvent = self.dragEnterEvent
        self.ui.centralwidget.dropEvent = self.dropEvent
        self.ui.centralwidget.dragMoveEvent = self.dragMoveEvent
        self.ui.centralwidget.mouseReleaseEvent = self.mouseReleaseEvent
        self.ui.centralwidget.mousePressEvent = self.mousePressEvent

        self.ui.centralwidget.setAcceptDrops(True)
        self.image = None


    def mouseMoveEvent(self, event):
        if self.image is not None:
            position = QPoint(event.x(), event.y())
            c = self.QImage_image1.pixel(event.x(), event.y())
            color = QColor.fromRgb(c).getRgb()
            txt = "(%d,%d)  (%d,%d,%d)"%(event.x(), event.y(), color[0], color[1], color[2])
            self.ui.RGBLabel.setText(txt)


    def show_image_in_Qlabel(self, image, viewer):
        self.QImage_image1 = QtGui.QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3, QtGui.QImage.Format_RGB888).scaled(viewer.width(), viewer.height())
        # self.QImage_image1 = QtGui.QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3, QtGui.QImage.Format_RGB888)
        self.QImage_image2 = QtGui.QPixmap(self.QImage_image1)
        # image2 = QtGui.QPixmap(img3)
        viewer.setPixmap(self.QImage_image2)


    def dragEnterEvent(self, evn):
        # self.ui.centralwidget.setWindowTitle('鼠标拖入窗口了')
        # print(evn.mimeData().text())
        # self.ui.RGBLabel.setText('文件路径：\n' + evn.mimeData().text())
        evn.accept()

    # 鼠标放开执行
    def dropEvent(self, evn):
        print()

        self.image = cv2.imread(evn.mimeData().text()[8:])
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        H, W = self.image.shape[0], self.image.shape[1]
        a, b = H, W
        while (b != 0):
            temp = a % b
            a = b
            b = temp
        temp = 600 // (H / a)
        view_H = (H / a) * temp
        view_W = (W / a) * temp
        # view_H, view_W = H, W

        self.ui.label.setGeometry(QRect(0, 0, int(view_W), int(view_H)))
        print(view_W, view_H)
        print(self.image.mean(axis=(0, 1)))
        self.show_image_in_Qlabel(self.image, self.ui.label)
        # self.ui.centralwidget.setWindowTitle('鼠标放开了')

    def dragMoveEvent(self, evn):

        return
        # print('鼠标移入')

    def mousePressEvent(self, event):
        print("clicked", event.pos().x(), event.pos().y())

    def mouseReleaseEvent(self, event):
        print("released", event.pos().x(), event.pos().y())

        # rect = QRect(50, 50, 100, 100)
        # painter = QPainter()
        # painter.setPen(QPen(Qt.red, 100, Qt.SolidLine))
        # painter.drawRect(rect)
        # self.update()