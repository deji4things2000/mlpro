<<<<<<< HEAD
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
=======
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter, QStatusBar, QLabel, QAction, QProgressDialog,
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QCheckBox, QComboBox, QPushButton, QMessageBox, QHBoxLayout
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

from .binary_explorer import BinaryExplorer
from .disassembly_view import DisassemblyView
from .llm_analysis import LLMAnalysisView

from core import cfg_builder, llm_analyzer, ghidra_bridge
from core import local_decompiler, translator
from core.disassembler import get_preview
from utils.file_utils import read_json
from utils.file_utils import write_json
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
        splitter.setChildrenCollapsible(False)

        self.binary_explorer = BinaryExplorer()
        splitter.addWidget(self.binary_explorer)

        self.disassembly_view = DisassemblyView()
        splitter.addWidget(self.disassembly_view)

        self.llm_analysis_view = LLMAnalysisView()
        splitter.addWidget(self.llm_analysis_view)

        splitter.setSizes([300, 500, 600])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 2)
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
        self._bridge_menu = bridge_menu
        connect_bridge_action = QAction("Connect to Bridge", self)
        connect_bridge_action.triggered.connect(self.connect_bridge)
        bridge_menu.addAction(connect_bridge_action)
        force_reconnect_action = QAction("Force Reconnect", self)
        force_reconnect_action.triggered.connect(self.force_reconnect)
        bridge_menu.addAction(force_reconnect_action)
        refresh_status_action = QAction("Refresh Status", self)
        refresh_status_action.triggered.connect(self.refresh_bridge_status)
        bridge_menu.addAction(refresh_status_action)

        # Settings menu
        settings_menu = menu_bar.addMenu("Settings")
        prefs_action = QAction("Preferences…", self)
        prefs_action.triggered.connect(self.open_settings)
        settings_menu.addAction(prefs_action)

        # Conditionally hide Bridge menu when bridge is disabled in config
        if not self.config.get("ghidra", {}).get("use_bridge"):
            try:
                bridge_menu.menuAction().setVisible(False)
            except Exception:
                pass

        self.binary_explorer.functions_tree.itemClicked.connect(self.on_function_selected)
        # Obsolete CFG/LLM/Ghidra buttons removed from LLMAnalysisView
        # Convert-to-Python button under decompiled view
        try:
            self.disassembly_view.convert_to_python_requested.connect(self.translate_to_python)
        except Exception:
            pass
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

    # --- RetDec worker with progress ---
    class RetdecWorker(QThread):
        finished_code = pyqtSignal(str)
        failed = pyqtSignal(str)

        def __init__(self, path: str, use_cached: bool = True):
            super().__init__()
            self.path = path
            self.use_cached = use_cached

        def run(self):
            try:
                if self.use_cached:
                    code = local_decompiler.decompile_with_retdec_cached(self.path)
                else:
                    code = local_decompiler.decompile_with_retdec(self.path)
                self.finished_code.emit(code)
            except Exception as e:
                self.failed.emit(str(e))

    def start_retdec_with_progress(self, path: str, use_cached: bool = True):
        if not local_decompiler.is_retdec_available():
            self.status_bar.showMessage("RetDec not found. Install via 'brew install retdec'.")
            return
        self.status_bar.showMessage("Decompiling with RetDec...")
        self._retdec_cancelled = False
        self._retdec_dialog = QProgressDialog("Decompiling with RetDec...", "Cancel", 0, 0, self)
        self._retdec_dialog.setWindowTitle("RetDec Decompilation")
        self._retdec_dialog.setModal(True)
        self._retdec_dialog.canceled.connect(self._on_retdec_canceled)
        self._retdec_dialog.show()

        self._retdec_worker = MainWindow.RetdecWorker(path, use_cached)
        self._retdec_worker.finished_code.connect(self._on_retdec_done)
        self._retdec_worker.failed.connect(self._on_retdec_failed)
        self._retdec_worker.start()

    def _on_retdec_canceled(self):
        self._retdec_cancelled = True
        self.status_bar.showMessage("RetDec canceled")

    def _on_retdec_done(self, code: str):
        try:
            if not getattr(self, "_retdec_cancelled", False):
                self.disassembly_view.set_decompiled_text(code)
                self.status_bar.showMessage("RetDec decompilation complete")
        finally:
            dlg = getattr(self, "_retdec_dialog", None)
            if dlg:
                dlg.close()

    def _on_retdec_failed(self, msg: str):
        try:
            if not getattr(self, "_retdec_cancelled", False):
                self.disassembly_view.set_decompiled_text(
                    f"// RetDec failed: {msg}\n// You can still use translation or Ghidra Bridge when available."
                )
                self.status_bar.showMessage(f"RetDec failed: {msg}")
        finally:
            dlg = getattr(self, "_retdec_dialog", None)
            if dlg:
                dlg.close()

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

    def decompile_with_retdec(self):
        path = self.current_binary
        if not path:
            self.status_bar.showMessage("Load a binary first")
            return
        # Run with progress (non-cached for manual action)
        self.start_retdec_with_progress(path, use_cached=False)

    def translate_to_python(self):
        code = self.disassembly_view.decomp_text.toPlainText()
        if not code.strip():
            self.status_bar.showMessage("No decompiled code to translate")
            return
        # Try LLM translation first if configured
        llm_cfg = self.config.get("llm", {})
        provider = llm_cfg.get("provider", "local")
        model = llm_cfg.get("model", "placeholder")
        api_key = llm_cfg.get("api_key") or None
        py = None
        try:
            from core import llm_analyzer as _la
            py = _la.translate_to_python(code, provider=provider, model=model, api_key=api_key)
        except Exception:
            py = None
        # Fallback to heuristic if LLM not available or failed
        if not py:
            py = translator.to_python(code)
        self.llm_analysis_view.update_analysis("Translated to Python", "N/A", "Review types and logic")
        self.disassembly_view.set_decompiled_text(py)
        self.status_bar.showMessage("Translation to Python complete")

    def translate_to_cpp(self):
        code = self.disassembly_view.decomp_text.toPlainText()
        if not code.strip():
            self.status_bar.showMessage("No decompiled code to translate")
            return
        cpp = translator.to_cpp(code)
        self.llm_analysis_view.update_analysis("Translated to C++", "N/A", "Refine types and classes")
        self.disassembly_view.set_decompiled_text(cpp)
        self.status_bar.showMessage("Translation to C++ complete")

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
            # Auto-run RetDec (cached) for local C/C++ pseudocode if available
            if local_decompiler.is_retdec_available():
                self.start_retdec_with_progress(path, use_cached=True)
            else:
                self.disassembly_view.set_decompiled_text(
                    "// RetDec not found. Install via 'brew install retdec' to enable local decompilation."
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

    def _cfg_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "config.json"

    def open_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Preferences")
        layout = QVBoxLayout(dlg)
        form = QFormLayout()

        gh = self.config.get("ghidra", {})
        llm = self.config.get("llm", {})

        use_bridge_cb = QCheckBox()
        use_bridge_cb.setChecked(bool(gh.get("use_bridge", False)))
        host_edit = QLineEdit(str(gh.get("host", "127.0.0.1")))
        port_edit = QLineEdit(str(gh.get("port", 18001)))
        install_edit = QLineEdit(str(gh.get("install_dir", "/Applications/ghidra/ghidra_12.0_PUBLIC")))

        provider_combo = QComboBox()
        provider_combo.addItems(["local", "openai", "ollama"])
        cur_provider = str(llm.get("provider", "local"))
        idx = provider_combo.findText(cur_provider)
        provider_combo.setCurrentIndex(idx if idx >= 0 else 0)
        model_edit = QLineEdit(str(llm.get("model", "placeholder")))
        api_key_edit = QLineEdit(str(llm.get("api_key", "")))

        form.addRow("Use Bridge", use_bridge_cb)
        form.addRow("Bridge Host", host_edit)
        form.addRow("Bridge Port", port_edit)
        form.addRow("Ghidra Install Dir", install_edit)
        form.addRow("LLM Provider", provider_combo)
        form.addRow("LLM Model", model_edit)
        form.addRow("LLM API Key", api_key_edit)

        layout.addLayout(form)
        btn_box = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        btn_box.addWidget(save_btn)
        btn_box.addWidget(cancel_btn)
        layout.addLayout(btn_box)

        def on_save():
            try:
                new_cfg = {
                    "theme": self.config.get("theme", "dark"),
                    "ghidra": {
                        "use_bridge": use_bridge_cb.isChecked(),
                        "host": host_edit.text().strip() or "127.0.0.1",
                        "port": int(port_edit.text().strip() or "18001"),
                        "install_dir": install_edit.text().strip() or "/Applications/ghidra/ghidra_12.0_PUBLIC",
                    },
                    "llm": {
                        "provider": provider_combo.currentText(),
                        "model": model_edit.text().strip() or "placeholder",
                        "api_key": api_key_edit.text().strip(),
                    },
                }
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Invalid settings: {e}")
                return
            try:
                write_json(self._cfg_path(), new_cfg)
                self.apply_config_updates(new_cfg)
                QMessageBox.information(self, "Saved", "Preferences saved.")
                dlg.accept()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save config: {e}")

        def on_cancel():
            dlg.reject()

        save_btn.clicked.connect(on_save)
        cancel_btn.clicked.connect(on_cancel)
        dlg.exec_()

    def apply_config_updates(self, new_cfg: dict):
        self.config = new_cfg
        gh = self.config.get("ghidra", {})
        # Update bridge connection
        try:
            ghidra_bridge.set_connection(str(gh.get("host", "127.0.0.1")), int(gh.get("port", 18001)))
        except Exception:
            pass
        # Show/hide Bridge menu
        try:
            self._bridge_menu.menuAction().setVisible(bool(gh.get("use_bridge", False)))
        except Exception:
            pass
        # Refresh status and optionally auto-populate
        self.refresh_bridge_status()
        if gh.get("use_bridge") and ghidra_bridge.is_available() and ghidra_bridge.has_current_program():
            self._auto_connect_and_populate()
>>>>>>> d779da83386b288f3c7dc115a1e68eb4253363d8
