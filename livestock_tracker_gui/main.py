import sys
from PyQt5 import QtWidgets
from livestock_tracker_gui.gui import MainWindow, init_db

def main():
    init_db()
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
