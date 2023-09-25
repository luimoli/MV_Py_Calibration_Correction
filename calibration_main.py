
import sys
from calibration.calibration import Ui_MainWindow
from PyQt5.QtWidgets import QApplication,QMainWindow
from calibration.calibration_function import CalibrationFunc
from PyQt5 import QtCore

if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

    app = QApplication(sys.argv)
    mainWnd =QMainWindow()
    ui = Ui_MainWindow()

    # 可以理解成将创建的 ui 绑定到新建的 mainWnd 上
    ui.setupUi(mainWnd)
    exposure_time = 40
    display = CalibrationFunc(ui, mainWnd, "calibration_3nh.npy", exposure_time, "raw")

    mainWnd.show()

    sys.exit(app.exec_())
