from PySide6.QtWidgets import QMainWindow, QSplitter, QWidget, QVBoxLayout
from PySide6.QtCore import Qt

from ui.binary_explorer import BinaryExplorer
from ui.disassembly_view import DisassemblyView
from ui.llm_analysis import LLMAnalysis
from core.ghidra_bridge import GhidraBridge
from core.llm_analyzer import analyze
from core.cfg_builder import build_cfg


class MainWindow(QMainWindow):
    def __init__(self, config, logger, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hybrid Ghidra GUI")
        self.resize(1200, 800)
        self.config = config
        self.logger = logger
        self.bridge = GhidraBridge(config, logger)

        splitter = QSplitter(Qt.Horizontal)
        self.binaryExplorer = BinaryExplorer(logger)
        self.disasmView = DisassemblyView()
        self.analysisView = LLMAnalysis()

        splitter.addWidget(self.binaryExplorer)
        splitter.addWidget(self.disasmView)
        splitter.addWidget(self.analysisView)
        splitter.setSizes([300, 600, 300])

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(splitter)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(container)

        self.binaryExplorer.binarySelected.connect(self.onBinarySelected)

    def onBinarySelected(self, path: str):
        self.logger.info(f"Selected binary: {path}")
        lines = self.bridge.disassemble(path)
        self.disasmView.setDisassembly(lines)
        cfg = build_cfg(lines)
        summary = analyze(lines, cfg)
        self.analysisView.setAnalysis(summary)
