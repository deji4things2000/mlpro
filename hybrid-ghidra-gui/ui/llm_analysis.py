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
