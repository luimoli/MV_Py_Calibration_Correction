
import sys
from correction.correction import Ui_MainWindow
from PyQt5.QtWidgets import QApplication,QMainWindow
from correction.correction_function import CorrectionFunc
from PyQt5 import QtCore

if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

    app = QApplication(sys.argv)
    mainWnd =QMainWindow()
    ui = Ui_MainWindow()

    # 可以理解成将创建的 ui 绑定到新建的 mainWnd 上
    ui.setupUi(mainWnd)
    exposure_time = 40
    display = CorrectionFunc(ui, mainWnd, "calibration_3nh.npy", exposure_time, "raw")
    mainWnd.show()

    sys.exit(app.exec_())
