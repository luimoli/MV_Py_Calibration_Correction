import sys
from PyQt5 import (QtWidgets, QtCore)
from PyQt5.QtWidgets import QHBoxLayout
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(603, 553)

        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.gridlayout = QtWidgets.QHBoxLayout(self.centralWidget)
        self.vtkWidget = QVTKRenderWindowInteractor(self.centralWidget)
        self.gridlayout.addWidget(self.vtkWidget)
        MainWindow.setCentralWidget(self.centralWidget)


class SimpleView(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ren = vtk.vtkRenderer()
        self.ui.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.ui.vtkWidget.GetRenderWindow().GetInteractor()

        # Create source
        source = vtk.vtkSphereSource()
        source.SetCenter(0, 0, 0)
        source.SetRadius(5.0)

        # vtkJPEGReader

        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())

        # Create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        self.ren.AddActor(actor)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SimpleView()
    window.show()
    window.iren.Initialize()  # Need this line to actually show the render inside Qt
    sys.exit(app.exec_())


# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
# from PyQt5.QtCore import *
# from PyQt5.QtOpenGL import QGLWidget
# import sys
# from vtk import *
# from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
# import vtk
#
#
# class MainWindow(QMainWindow):
#     """docstring for Mainwindow"""
#
#     def __init__(self, parent=None):
#         super(MainWindow, self).__init__(parent)
#         self.basic()
#         splitter_main = self.split_()
#         self.setCentralWidget(splitter_main)
#
#         # 窗口基础属性
#
#     def basic(self):
#         # 设置标题，大小，图标
#         self.setWindowTitle("GT")
#         self.resize(1100, 650)
#         self.setWindowIcon(QIcon("./image/Gt.png"))
#         # 居中显示
#         screen = QDesktopWidget().geometry()
#         self_size = self.geometry()
#         self.move((screen.width() - self_size.width()) / 2, (screen.height() - self_size.height()) / 2)
#
#         # 分割窗口
#
#     def split_(self):
#         splitter = QSplitter(Qt.Vertical)
#         frame = QFrame()
#         vl = QVBoxLayout()
#         vtkWidget = QVTKRenderWindowInteractor()
#         vl.addWidget(vtkWidget)
#         # vl.setContentsMargins(0,0,0,0)
#         ren = vtk.vtkRenderer()
#         vtkWidget.GetRenderWindow().AddRenderer(ren)
#         self.iren = vtkWidget.GetRenderWindow().GetInteractor()
#         self.CreateCone(ren)
#         frame.setLayout(vl)
#         splitter.addWidget(frame)
#         testedit = QTextEdit()
#         splitter.addWidget(testedit)
#         splitter.setStretchFactor(0, 3)
#         splitter.setStretchFactor(1, 2)
#         splitter_main = QSplitter(Qt.Horizontal)
#         textedit_main = QTextEdit()
#         splitter_main.addWidget(textedit_main)
#         splitter_main.addWidget(splitter)
#         splitter_main.setStretchFactor(0, 2)
#         splitter_main.setStretchFactor(1, 5)
#         return splitter_main
#
#     def CreateCone(self, ren):
#         # Create source
#         source = vtk.vtkSphereSource()
#         source.SetCenter(0, 0, 0)
#         source.SetRadius(5.0)
#
#         # Create a mapper
#         mapper = vtk.vtkPolyDataMapper()
#         mapper.SetInputConnection(source.GetOutputPort())
#
#         # Create an actor
#         actor = vtk.vtkActor()
#         actor.SetMapper(mapper)
#
#         ren.AddActor(actor)
#         ren.ResetCamera()
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     win = MainWindow()
#     win.show()
#     win.iren.Initialize()
#     sys.exit(app.exec_())
