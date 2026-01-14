<<<<<<< HEAD
from PySide6.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor
from PySide6.QtCore import QRegularExpression


class SimpleAsmHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.addressFormat = QTextCharFormat()
        self.addressFormat.setForeground(QColor("#888"))
        self.opcodeFormat = QTextCharFormat()
        self.opcodeFormat.setForeground(QColor("#005cc5"))
        self.commentFormat = QTextCharFormat()
        self.commentFormat.setForeground(QColor("#6a737d"))

        self.addressRe = QRegularExpression(r"^\s*[0-9a-fA-F]+:")
        self.opcodeRe = QRegularExpression(r"\b(mov|cmp|call|jmp|ret|push|pop|add|sub|lea)\b")
        self.commentRe = QRegularExpression(r";.*$")

    def highlightBlock(self, text: str):
        m = self.addressRe.match(text)
        if m.hasMatch():
            start = m.capturedStart()
            length = m.capturedLength()
            self.setFormat(start, length, self.addressFormat)

        it = self.opcodeRe.globalMatch(text)
        while it.hasNext():
            m = it.next()
            self.setFormat(m.capturedStart(), m.capturedLength(), self.opcodeFormat)

        it = self.commentRe.globalMatch(text)
        while it.hasNext():
            m = it.next()
            self.setFormat(m.capturedStart(), m.capturedLength(), self.commentFormat)
=======
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
>>>>>>> d779da83386b288f3c7dc115a1e68eb4253363d8
