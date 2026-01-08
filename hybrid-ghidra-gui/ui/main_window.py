from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter, QStatusBar
)
from PyQt5.QtCore import Qt

from .binary_explorer import BinaryExplorer
from .disassembly_view import DisassemblyView
from .llm_analysis import LLMAnalysisView

from core import cfg_builder, llm_analyzer, ghidra_bridge


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Hybrid Ghidra Python GUI')
        self.setGeometry(100, 100, 1400, 900)
        self.current_binary: str | None = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        splitter = QSplitter(Qt.Horizontal)

        self.binary_explorer = BinaryExplorer()
        splitter.addWidget(self.binary_explorer)

        self.disassembly_view = DisassemblyView()
        splitter.addWidget(self.disassembly_view)

        self.llm_analysis_view = LLMAnalysisView()
        splitter.addWidget(self.llm_analysis_view)

        splitter.setSizes([300, 500, 600])
        main_layout.addWidget(splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        self.binary_explorer.functions_tree.itemClicked.connect(self.on_function_selected)
        self.llm_analysis_view.build_cfg_btn.clicked.connect(self.build_cfg)
        self.llm_analysis_view.run_llm_btn.clicked.connect(self.run_llm_analysis)
        self.llm_analysis_view.annotate_btn.clicked.connect(self.annotate_in_ghidra)
        self.binary_explorer.status_message.connect(self.status_bar.showMessage)
        self.binary_explorer.binary_loaded.connect(self.on_binary_loaded)

        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #2b2b2b;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #569CD6;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #555;
            }
            QTreeWidget {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #555;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QLabel {
                color: #d4d4d4;
            }
            QCheckBox {
                color: #d4d4d4;
            }
            """
        )

    def on_function_selected(self, item, column):
        function_name = item.text(0)
        self.status_bar.showMessage(f"Selected function: {function_name}")

        result = llm_analyzer.analyze_function(function_name)
        self.llm_analysis_view.update_analysis(
            result.get("summary", "No summary"),
            result.get("prediction", "UNKNOWN"),
            result.get("fix", "No suggestion"),
        )

    def build_cfg(self):
        self.status_bar.showMessage("Building Control Flow Graph...")
        graph_text = cfg_builder.build_cfg()
        self.llm_analysis_view.cfg_text.setPlainText(graph_text)

    def run_llm_analysis(self):
        self.status_bar.showMessage("Running LLM Analysis...")
        item = self.binary_explorer.functions_tree.currentItem()
        function_name = item.text(0) if item else "main"
        result = llm_analyzer.analyze_function(function_name)
        self.llm_analysis_view.update_analysis(
            result.get("summary", "No summary"),
            result.get("prediction", "UNKNOWN"),
            result.get("fix", "No suggestion"),
        )

    def annotate_in_ghidra(self):
        self.status_bar.showMessage("Annotating in Ghidra...")
        item = self.binary_explorer.functions_tree.currentItem()
        function_name = item.text(0) if item else "main"
        success = ghidra_bridge.annotate(function_name, note="Annotated by Hybrid GUI")
        if success:
            self.status_bar.showMessage("Annotation complete")
        else:
            self.status_bar.showMessage("Annotation failed")

    def on_binary_loaded(self, path: str):
        self.current_binary = path
        self.status_bar.showMessage(f"Loaded: {path}")
        # Update views with basic placeholders tied to the binary path
        self.disassembly_view.show_loaded_binary(path)
        self.llm_analysis_view.reset_for_binary(path)
        # Populate functions with a path-derived sample for immediate UX feedback
        base = path.split('/')[-1]
        sample_funcs = [
            (f"start_{base}", "0x401000", "Low"),
            (f"main_{base}", "0x401234", "Medium"),
            (f"parse_input_{base}", "0x401567", "High"),
        ]
        self.binary_explorer.set_functions(sample_funcs)
        # Auto-select first function to show analysis immediately
        first = self.binary_explorer.functions_tree.topLevelItem(0)
        if first:
            self.binary_explorer.functions_tree.setCurrentItem(first)
            self.on_function_selected(first, 0)
