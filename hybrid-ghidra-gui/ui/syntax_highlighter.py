from PyQt5.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor
from PyQt5.QtCore import QRegExp


class Highlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        assembly_format = QTextCharFormat()
        assembly_format.setForeground(QColor("#569CD6"))
        self.highlighting_rules.append((r"\b(mov|push|pop|call|jmp|ret|add|sub|cmp|test)\b", assembly_format))

        address_format = QTextCharFormat()
        address_format.setForeground(QColor("#CE9178"))
        self.highlighting_rules.append((r"\b[0-9A-F]{8,16}\b", address_format))

        register_format = QTextCharFormat()
        register_format.setForeground(QColor("#9CDCFE"))
        self.highlighting_rules.append((r"\b(rax|rbx|rcx|rdx|rsp|rbp|rsi|rdi|r8|r9|r10|r11|r12|r13|r14|r15|eax|ebx|ecx|edx)\b", register_format))

    def highlightBlock(self, text: str) -> None:
        for pattern, fmt in self.highlighting_rules:
            expr = QRegExp(pattern)
            index = expr.indexIn(text, 0)
            while index >= 0:
                length = expr.matchedLength()
                self.setFormat(index, length, fmt)
                index = expr.indexIn(text, index + length)
