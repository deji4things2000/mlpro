from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QPushButton, QLabel,
    QTreeWidget, QTreeWidgetItem, QCheckBox
)
from PyQt5.QtCore import pyqtSignal


class BinaryExplorer(QWidget):
    status_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        load_group = QGroupBox("Binary Explorer")
        load_layout = QVBoxLayout()

        self.load_btn = QPushButton("Load Binary")
        self.load_btn.clicked.connect(self.load_binary)

        self.binary_path_label = QLabel("No binary loaded")

        load_layout.addWidget(self.load_btn)
        load_layout.addWidget(self.binary_path_label)
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)

        functions_group = QGroupBox("Functions List")
        functions_layout = QVBoxLayout()

        self.functions_tree = QTreeWidget()
        self.functions_tree.setHeaderLabels(["Function", "Address", "Risk Level"])

        sample_functions = [
            ("start", "0x401000", "Low"),
            ("main", "0x401234", "Medium"),
            ("sub_401234", "0x401234", "High"),
            ("parse_input", "0x401567", "High"),
            ("check_auth", "0x401600", "Medium"),
        ]

        for func, addr, risk in sample_functions:
            item = QTreeWidgetItem([func, addr, risk])
            self.functions_tree.addTopLevelItem(item)

        functions_layout.addWidget(self.functions_tree)
        functions_group.setLayout(functions_layout)
        layout.addWidget(functions_group)

        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout()

        self.high_risk_check = QCheckBox("High Risk")
        self.medium_risk_check = QCheckBox("Medium Risk")
        self.low_risk_check = QCheckBox("Low Risk")

        self.high_risk_check.setChecked(True)
        self.medium_risk_check.setChecked(True)
        self.low_risk_check.setChecked(True)

        filter_layout.addWidget(self.high_risk_check)
        filter_layout.addWidget(self.medium_risk_check)
        filter_layout.addWidget(self.low_risk_check)

        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        self.setLayout(layout)

    def load_binary(self):
        self.binary_path_label.setText("/path/to/binary.exe")
        self.status_message.emit("Binary loaded successfully")
