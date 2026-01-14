<<<<<<< HEAD
import json
import os
import sys

from utils.logger import get_logger
from utils.file_utils import ensure_config_defaults

logger = get_logger("hybrid-ghidra-gui")


def load_config(path):
    try:
        with open(path, 'r') as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {}
    return ensure_config_defaults(cfg)


def main():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "config.json"))
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as e:
        logger.error("PySide6 not available. Install dependencies: %s", e)
        print("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)

    from ui.main_window import MainWindow
    app = QApplication(sys.argv)
    win = MainWindow(cfg, logger)
    win.show()
    sys.exit(app.exec())
=======
import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
>>>>>>> d779da83386b288f3c7dc115a1e68eb4253363d8


if __name__ == "__main__":
    main()
