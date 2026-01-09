from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter, QStatusBar, QLabel, QAction
)
from PyQt5.QtCore import Qt, QTimer

from .binary_explorer import BinaryExplorer
from .disassembly_view import DisassemblyView
from .llm_analysis import LLMAnalysisView

from core import cfg_builder, llm_analyzer, ghidra_bridge
from core.disassembler import get_preview
from utils.file_utils import read_json
from pathlib import Path
import sys
import subprocess
import os


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Hybrid Ghidra Python GUI')
        self.setGeometry(100, 100, 1400, 900)
        self.current_binary: str | None = None
        # Load config
        cfg_path = Path(__file__).resolve().parents[1] / "config.json"
        try:
            self.config = read_json(cfg_path)
        except Exception:
            self.config = {"ghidra": {"use_bridge": False}, "llm": {"provider": "local", "model": "placeholder", "api_key": ""}}
        # Configure bridge connection if present
        gh_cfg = self.config.get("ghidra", {})
        host = gh_cfg.get("host", "127.0.0.1")
        port = int(gh_cfg.get("port", 18001))
        try:
            ghidra_bridge.set_connection(host, port)
        except Exception:
            pass

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
        # Bridge status indicator
        self.bridge_status_label = QLabel("Bridge: Unknown")
        self.status_bar.addPermanentWidget(self.bridge_status_label)
        # Auto-refresh bridge status periodically to reflect late starts
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(2000)
        self._status_timer.timeout.connect(self.refresh_bridge_status)
        self._status_timer.start()
        # Attempt to auto-populate functions from an already running bridge
        QTimer.singleShot(500, self._auto_connect_and_populate)

        # Bridge menu
        menu_bar = self.menuBar()
        bridge_menu = menu_bar.addMenu("Bridge")
        connect_bridge_action = QAction("Connect to Bridge", self)
        connect_bridge_action.triggered.connect(self.connect_bridge)
        bridge_menu.addAction(connect_bridge_action)
        force_reconnect_action = QAction("Force Reconnect", self)
        force_reconnect_action.triggered.connect(self.force_reconnect)
        bridge_menu.addAction(force_reconnect_action)
        refresh_status_action = QAction("Refresh Status", self)
        refresh_status_action.triggered.connect(self.refresh_bridge_status)
        bridge_menu.addAction(refresh_status_action)

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
        # Initial bridge status
        self.refresh_bridge_status()

    def on_function_selected(self, item, column):
        function_name = item.text(0)
        self.status_bar.showMessage(f"Selected function: {function_name}")
        asm = ""
        decomp = ""
        if self.config.get("ghidra", {}).get("use_bridge") and ghidra_bridge.is_available():
            asm = ghidra_bridge.get_disassembly(function_name)
            decomp = ghidra_bridge.get_decompiled(function_name)
            if asm:
                self.disassembly_view.set_assembly_text(asm)
            if decomp:
                self.disassembly_view.set_decompiled_text(decomp)
        result = llm_analyzer.analyze_function(
            function_name,
            assembly=asm,
            decompiled=decomp,
            provider=self.config.get("llm", {}).get("provider", "local"),
            model=self.config.get("llm", {}).get("model", "placeholder"),
            api_key=self.config.get("llm", {}).get("api_key") or None,
        )
        self.llm_analysis_view.update_analysis(
            result.get("summary", "No summary"),
            result.get("prediction", "UNKNOWN"),
            result.get("fix", "No suggestion"),
        )

    def build_cfg(self):
        self.status_bar.showMessage("Building Control Flow Graph...")
        item = self.binary_explorer.functions_tree.currentItem()
        function_name = item.text(0) if item else "main"
        graph_text = cfg_builder.build_cfg(function_name=function_name)
        self.llm_analysis_view.cfg_text.setPlainText(graph_text)
        self.refresh_bridge_status()

    def run_llm_analysis(self):
        self.status_bar.showMessage("Running LLM Analysis...")
        item = self.binary_explorer.functions_tree.currentItem()
        function_name = item.text(0) if item else "main"
        asm = self.disassembly_view.asm_text.toPlainText()
        decomp = self.disassembly_view.decomp_text.toPlainText()
        result = llm_analyzer.analyze_function(
            function_name,
            assembly=asm,
            decompiled=decomp,
            provider=self.config.get("llm", {}).get("provider", "local"),
            model=self.config.get("llm", {}).get("model", "placeholder"),
            api_key=self.config.get("llm", {}).get("api_key") or None,
        )
        self.llm_analysis_view.update_analysis(
            result.get("summary", "No summary"),
            result.get("prediction", "UNKNOWN"),
            result.get("fix", "No suggestion"),
        )

    def annotate_in_ghidra(self):
        self.status_bar.showMessage("Annotating in Ghidra...")
        item = self.binary_explorer.functions_tree.currentItem()
        function_name = item.text(0) if item else "main"
        success = False
        if self.config.get("ghidra", {}).get("use_bridge") and ghidra_bridge.is_available():
            success = ghidra_bridge.annotate(function_name, note="Annotated by Hybrid GUI")
        if success:
            self.status_bar.showMessage("Annotation complete")
        else:
            self.status_bar.showMessage("Annotation failed")
        self.refresh_bridge_status()

    def on_binary_loaded(self, path: str):
        self.current_binary = path
        self.status_bar.showMessage(f"Loaded: {path}")
        self.refresh_bridge_status()
        # If configured and available, open in Ghidra and fetch functions
        used_bridge = False
        funcs = []
        if self.config.get("ghidra", {}).get("use_bridge") and ghidra_bridge.is_available():
            # Best-effort: if a program is already open, we'll get functions; otherwise fallback
            used_bridge = ghidra_bridge.open_program(path)
            funcs = ghidra_bridge.list_functions()
        if funcs:
            self.binary_explorer.set_functions(funcs)
        else:
            # Fallback: disassemble locally and set functions/preview
            asm, funcs = get_preview(path)
            self.binary_explorer.set_functions(funcs)
            self.disassembly_view.set_assembly_text(f"; Disassembly (local preview)\n{asm}")
            self.disassembly_view.set_decompiled_text(
                "// Decompiled preview requires Ghidra Bridge or analyzer integration."
            )
            self.llm_analysis_view.reset_for_binary(path)
        # Auto-select first function
        first = self.binary_explorer.functions_tree.topLevelItem(0)
        if first:
            self.binary_explorer.functions_tree.setCurrentItem(first)
            self.on_function_selected(first, 0)

    def refresh_bridge_status(self):
        use_bridge = self.config.get("ghidra", {}).get("use_bridge")
        if not use_bridge:
            self.bridge_status_label.setText("Bridge: Off")
            self.bridge_status_label.setStyleSheet("color: #aaaaaa")
            return
        connected = ghidra_bridge.is_available()
        if connected:
            self.bridge_status_label.setText("Bridge: Connected")
            self.bridge_status_label.setStyleSheet("color: #4CAF50")
        else:
            self.bridge_status_label.setText("Bridge: Disconnected")
            self.bridge_status_label.setStyleSheet("color: #f44336")

    def _auto_connect_and_populate(self):
        # If a bridge server and program are already available, use them
        if not self.config.get("ghidra", {}).get("use_bridge"):
            return
        if not ghidra_bridge.is_available():
            return
        if not ghidra_bridge.has_current_program():
            return
        funcs = ghidra_bridge.list_functions()
        if funcs:
            self.binary_explorer.set_functions(funcs)
            prog = ghidra_bridge.get_program_name()
            if prog:
                self.status_bar.showMessage(f"Connected to Ghidra: {prog}")
            # Auto-select first function and display using bridge
            first = self.binary_explorer.functions_tree.topLevelItem(0)
            if first:
                self.binary_explorer.functions_tree.setCurrentItem(first)
                self.on_function_selected(first, 0)

    def connect_bridge(self):
        self.status_bar.showMessage("Connecting to Ghidra Bridge...")
        # Trigger lazy connect and update UI
        connected = ghidra_bridge.is_available()
        self.refresh_bridge_status()
        if connected:
            self._auto_connect_and_populate()
            self.status_bar.showMessage("Connected to Ghidra Bridge")
        else:
            self.status_bar.showMessage("Unable to connect. Start the bridge in Ghidra.")

    def force_reconnect(self):
        self.status_bar.showMessage("Forcing bridge reconnect...")
        gh = self.config.get("ghidra", {})
        host = gh.get("host", "127.0.0.1")
        port = int(gh.get("port", 18001))
        # Reset cached bridge so next call re-attempts connection
        try:
            ghidra_bridge.set_connection(host, port)
        except Exception:
            pass
        connected = ghidra_bridge.is_available()
        self.refresh_bridge_status()
        if connected and ghidra_bridge.has_current_program():
            self._auto_connect_and_populate()
            self.status_bar.showMessage("Reconnected to existing bridge")
        else:
            self.status_bar.showMessage("Reconnect attempted. Start bridge in Ghidra if needed.")
