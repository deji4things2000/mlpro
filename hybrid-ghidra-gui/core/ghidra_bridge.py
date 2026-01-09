from __future__ import annotations

from typing import Optional, List, Tuple
import os

_bridge = None
_conn_host = os.environ.get("GHIDRA_BRIDGE_HOST", "127.0.0.1")
_conn_port = int(os.environ.get("GHIDRA_BRIDGE_PORT", "18001"))


def set_connection(host: str, port: int) -> None:
    """Set the host/port used to connect to Ghidra Bridge."""
    global _conn_host, _conn_port, _bridge
    _conn_host = host
    _conn_port = port
    _bridge = None  # force reconnect with new params


def _get_bridge():
    global _bridge
    if _bridge is not None:
        return _bridge
    try:
        from ghidra_bridge import GhidraBridge
        _bridge = GhidraBridge(namespace={}, connect_to_host=_conn_host, connect_to_port=_conn_port)
        return _bridge
    except Exception:
        return None


def is_available() -> bool:
    return _get_bridge() is not None


def has_current_program() -> bool:
    br = _get_bridge()
    if br is None:
        return False
    try:
        return bool(br.remote_eval('currentProgram is not None'))
    except Exception:
        return False


def get_program_name() -> str:
    br = _get_bridge()
    if br is None:
        return ""
    try:
        name = br.remote_eval("currentProgram.getName() if currentProgram else ''")
        return name or ""
    except Exception:
        return ""


def open_program(path: str) -> bool:
    """Check if a program is open; importing via bridge is version-dependent.
    Returns True if a current program exists. """
    br = _get_bridge()
    if br is None:
        return False
    try:
        # Avoid relying on API differences; simply check for currentProgram
        has_prog = br.remote_eval('currentProgram is not None')
        return bool(has_prog)
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
        expr = "[(f.getName(), str(f.getEntryPoint()), 'Unknown') for f in list(getFunctionManager().getFunctions(True))]"
        result = br.remote_eval(expr)
        return list(result) if result else []
    except Exception:
        return []


def get_disassembly(function_name: str) -> str:
    br = _get_bridge()
    if br is None:
        return ""
    try:
        # Build lines with Python 2.7 compatible string formatting on Ghidra side
        expr = (
            "['%s  %s' % (str(ins.getAddress()), str(ins)) for ins in list("
            "currentProgram.getListing().getInstructions("
            "next((f for f in list(getFunctionManager().getFunctions(True)) if f.getName()==%r), None).getBody(), True))]"
            % function_name
        )
        lines = br.remote_eval(expr)
        return "\n".join(lines) if lines else ""
    except Exception:
        return ""


def get_decompiled(function_name: str) -> str:
    br = _get_bridge()
    if br is None:
        return ""
    try:
        code = (
            "from ghidra.app.decompiler import DecompInterface\n"
            "fm = currentProgram.getFunctionManager()\n"
            "func = next((f for f in fm.getFunctions(True) if f.getName()==%r), None)\n"
            "__decomp_text = ''\n"
            "if func:\n"
            "    ifc = DecompInterface()\n"
            "    ifc.openProgram(currentProgram)\n"
            "    res = ifc.decompileFunction(func, 60, monitor)\n"
            "    if res and res.getDecompiledFunction():\n"
            "        __decomp_text = res.getDecompiledFunction().getC()\n"
        ) % function_name
        br.remote_exec(code)
        text = br.remote_eval("__decomp_text")
        return text or ""
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
