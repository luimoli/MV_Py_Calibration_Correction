
import sys
from viewer.viewer_ui import Ui_MainWindow
from PyQt5.QtWidgets import QApplication,QMainWindow
from viewer.viewer_function import ViewerFunc
from PyQt5 import QtCore

if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

    app = QApplication(sys.argv)
    mainWnd = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWnd)
    exposure_time = 40
    display = ViewerFunc(ui, mainWnd)
    mainWnd.show()

    sys.exit(app.exec_())
