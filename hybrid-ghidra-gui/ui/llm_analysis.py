<<<<<<< HEAD
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextBrowser


class LLMAnalysis(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.title = QLabel("LLM Analysis")
        self.view = QTextBrowser()
        layout.addWidget(self.title)
        layout.addWidget(self.view)

    def setAnalysis(self, text: str):
        self.view.setText(text or "No analysis.")
=======
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QTextEdit, QSplitter
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt


class LLMAnalysisView(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        splitter = QSplitter()
        splitter.setOrientation(Qt.Vertical)

        analysis_group = QGroupBox("LLM Analysis")
        analysis_layout = QVBoxLayout()

        self.summary_label = QLabel("Summary: ")
        self.prediction_label = QLabel("Prediction: ")
        self.fix_label = QLabel("Suggested Fix: ")

        self.update_analysis(
            "Potential buffer overflow detected.",
            "HIGH RISK FUNCTION",
            "Sanitize input length",
        )

        analysis_layout.addWidget(self.summary_label)
        analysis_layout.addWidget(self.prediction_label)
        analysis_layout.addWidget(self.fix_label)

        cfg_group = QGroupBox("Control Flow Graph")
        cfg_layout = QVBoxLayout()

        self.cfg_text = QTextEdit()
        self.cfg_text.setFont(QFont("Courier", 10))
        self.cfg_text.setReadOnly(True)

        cfg_sample = (
            """main\n"
            "├── parse_input\n"
            "│   ├── validate_user\n"
            "│   └── error_exit\n"
            "└── crypto_routine\n"
            "    ├── encrypt_data\n"
            "    └── hash_password"""
        )

        self.cfg_text.setPlainText(cfg_sample)
        cfg_layout.addWidget(self.cfg_text)

        cfg_group.setLayout(cfg_layout)

        analysis_group.setLayout(analysis_layout)
        cfg_group.setLayout(cfg_layout)

        splitter.addWidget(analysis_group)
        splitter.addWidget(cfg_group)
        splitter.setSizes([200, 400])
        layout.addWidget(splitter)
        self.setLayout(layout)

    def update_analysis(self, summary: str, prediction: str, fix: str):
        self.summary_label.setText(f"<b>Summary:</b> {summary}")
        self.prediction_label.setText(f"<b>Prediction:</b> {prediction}")
        self.fix_label.setText(f"<b>Suggested Fix:</b> {fix}")

    def reset_for_binary(self, path: str) -> None:
        self.update_analysis(
            summary=f"Binary loaded: {path}",
            prediction="N/A",
            fix="Select a function or run analysis",
        )
        # Reset CFG text to a default prompt
        self.cfg_text.setPlainText(
            "(CFG will appear here after building)"
        )
>>>>>>> d779da83386b288f3c7dc115a1e68eb4253363d8
