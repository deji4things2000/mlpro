from PySide6.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit, QLabel

from ui.syntax_highlighter import SimpleAsmHighlighter


class DisassemblyView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.title = QLabel("Disassembly")
        self.editor = QPlainTextEdit()
        self.editor.setReadOnly(True)
        self.highlighter = SimpleAsmHighlighter(self.editor.document())

        layout.addWidget(self.title)
        layout.addWidget(self.editor)

    def setDisassembly(self, lines):
        text = "\n".join(lines or [])
        self.editor.setPlainText(text)
