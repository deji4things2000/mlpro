from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel
from PySide6.QtCore import Signal


class BinaryExplorer(QWidget):
    binarySelected = Signal(str)

    def __init__(self, logger, parent=None):
        super().__init__(parent)
        self.logger = logger
        self.setMinimumWidth(250)
        layout = QVBoxLayout(self)
        self.statusLabel = QLabel("No file selected")
        self.openBtn = QPushButton("Open Binary...")
        self.openBtn.clicked.connect(self.open_file)

        layout.addWidget(self.statusLabel)
        layout.addWidget(self.openBtn)
        layout.addStretch()

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Binary or Object", "", "All Files (*)")
        if path:
            self.statusLabel.setText(path)
            self.logger.info(f"Opened: {path}")
            self.binarySelected.emit(path)
