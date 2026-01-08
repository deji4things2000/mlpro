from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QTextEdit
from PyQt5.QtGui import QFont

from .syntax_highlighter import Highlighter


class DisassemblyView(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        asm_group = QGroupBox("Assembly")
        asm_layout = QVBoxLayout()

        self.asm_text = QTextEdit()
        self.asm_text.setFont(QFont("Courier", 10))
        self.highlighter = Highlighter(self.asm_text.document())

        sample_asm = (
            """00401234  55                 push rbp\n"
            "00401235  4889E5             mov rbp, rsp\n"
            "00401238  4889EC             mov rbx, rcx\n"
            "0040123B  E8 56 78 00 00     call sub_401567\n"
            "00401240  4885C0             test rax, rax\n"
            "00401243  740E               je 0x401253\n"
            "00401245  48C745F800000000   mov qword [rbp-0x8], 0x0"""
        )

        self.asm_text.setPlainText(sample_asm)
        asm_layout.addWidget(self.asm_text)
        asm_group.setLayout(asm_layout)
        layout.addWidget(asm_group)

        decomp_group = QGroupBox("Decompiled Code")
        decomp_layout = QVBoxLayout()

        self.decomp_text = QTextEdit()
        self.decomp_text.setFont(QFont("Courier", 10))

        sample_decomp = (
            """void parse_input(char *input) {\n"
            "    if (check_auth(input)) {\n"
            "        printf(\"Access Granted\\n\");\n"
            "    } else {\n"
            "        printf(\"Access Denied\\n\");\n"
            "    }\n"
            "}"""
        )

        self.decomp_text.setPlainText(sample_decomp)
        decomp_layout.addWidget(self.decomp_text)
        decomp_group.setLayout(decomp_layout)
        layout.addWidget(decomp_group)

        self.setLayout(layout)

    def set_assembly_text(self, text: str) -> None:
        self.asm_text.setPlainText(text)

    def set_decompiled_text(self, text: str) -> None:
        self.decomp_text.setPlainText(text)

    def show_loaded_binary(self, path: str) -> None:
        self.set_assembly_text(
            f"; Loaded binary: {path}\n; Disassembly preview is placeholder until integration"
        )
        self.set_decompiled_text(
            "// Decompiled preview not available without analyzer.\n"
            "// Use Run LLM Analysis or integrate Ghidra Bridge."
        )
