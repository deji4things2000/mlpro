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
