from __future__ import annotations

from typing import Optional, List, Tuple

_bridge = None


def _get_bridge():
    global _bridge
    if _bridge is not None:
        return _bridge
    try:
        from ghidra_bridge import GhidraBridge
        _bridge = GhidraBridge(namespace={})
        return _bridge
    except Exception:
        return None


def is_available() -> bool:
    return _get_bridge() is not None


def open_program(path: str) -> bool:
    """Attempt to open/import a program in Ghidra via the bridge.
    Returns True if successful, False otherwise.
    """
    br = _get_bridge()
    if br is None:
        return False
    try:
        # Execute a small script in Ghidra to import/open the program
        # Note: This requires the bridge server running inside a Ghidra session.
        script = f"""
from ghidra.util.task import TaskMonitor
from ghidra.app.util.opinion import AutoImporter
from ghidra.program.model.listing import Program
from java.io import File

f = File(r"{path}")
program = AutoImporter.importByUsingBestGuess(f, None, TaskMonitor.DUMMY)
if program is not None:
    state.getTool().addProgram(program)
    currentProgram = program
        """
        br.ghidra_script(script)
        return True
    except Exception:
        return False


def list_functions() -> List[Tuple[str, str, str]]:
    """Return a list of (name, address, risk) from the current program.
    If bridge not available, return an empty list.
    """
    br = _get_bridge()
    if br is None:
        return []
    try:
        script = """
funcs = list(getFunctionManager().getFunctions(True))
out = [(f.getName(), str(f.getEntryPoint()), "Unknown") for f in funcs]
        """
        result = br.evaluate(script, out_name="out")
        return list(result) if result else []
    except Exception:
        return []


def get_disassembly(function_name: str) -> str:
    br = _get_bridge()
    if br is None:
        return ""
    try:
        script = f"""
from ghidra.program.model.listing import Function
fm = getFunctionManager()
func = fm.getFunction("{function_name}")
text = ""
if func:
    code = currentProgram.getListing()
    it = code.getInstructions(func.getBody(), True)
    lines = []
    while it.hasNext():
        ins = it.next()
        lines.append(f"{ins.getAddress()}  {ins}")
    text = "\n".join(lines)
out = text
        """
        return br.evaluate(script, out_name="out") or ""
    except Exception:
        return ""


def get_decompiled(function_name: str) -> str:
    br = _get_bridge()
    if br is None:
        return ""
    try:
        script = f"""
from ghidra.app.decompiler import DecompInterface
fm = getFunctionManager()
func = fm.getFunction("{function_name}")
text = ""
if func:
    ifc = DecompInterface()
    ifc.openProgram(currentProgram)
    res = ifc.decompileFunction(func, 60, monitor)
    if res and res.getDecompiledFunction():
        text = res.getDecompiledFunction().getC()
out = text
        """
        return br.evaluate(script, out_name="out") or ""
    except Exception:
        return ""


def annotate(function_name: str, note: Optional[str] = None) -> bool:
    br = _get_bridge()
    if br is None:
        return False
    try:
        script = f"""
fm = getFunctionManager()
func = fm.getFunction("{function_name}")
if func:
    func.setComment("{note or ''}")
    out = True
else:
    out = False
        """
        result = br.evaluate(script, out_name="out")
        return bool(result)
    except Exception:
        return False
