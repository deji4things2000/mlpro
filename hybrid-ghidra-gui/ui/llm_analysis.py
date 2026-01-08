from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QTextEdit, QHBoxLayout, QPushButton
)
from PyQt5.QtGui import QFont


class LLMAnalysisView(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

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

        button_layout = QHBoxLayout()
        self.build_cfg_btn = QPushButton("Build CFG")
        self.run_llm_btn = QPushButton("Run LLM Analysis")
        self.annotate_btn = QPushButton("Annotate in Ghidra")

        button_layout.addWidget(self.build_cfg_btn)
        button_layout.addWidget(self.run_llm_btn)
        button_layout.addWidget(self.annotate_btn)
        cfg_layout.addLayout(button_layout)

        cfg_group.setLayout(cfg_layout)

        layout.addWidget(analysis_group)
        layout.addWidget(cfg_group)

        analysis_group.setLayout(analysis_layout)
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
