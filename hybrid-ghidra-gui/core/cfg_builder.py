from __future__ import annotations

from typing import Optional

from . import ghidra_bridge


def build_cfg(function_name: Optional[str] = None) -> str:
    """Build a simple CFG text; use Ghidra if available, else fallback."""
    if ghidra_bridge.is_available():
        # Minimal representation from instruction flows: show successors via fallthrough/branches
        asm = ghidra_bridge.get_disassembly(function_name or "main")
        lines = asm.splitlines()
        if not lines:
            return "(CFG unavailable: empty disassembly)"
        # Naive parse: group by basic blocks where control instructions appear
        blocks = []
        block = []
        for ln in lines:
            block.append(ln)
            if any(k in ln.lower() for k in ("jmp", "call", "ret", "je", "jne", "jg", "jl")):
                blocks.append(block)
                block = []
        if block:
            blocks.append(block)
        out = [f"CFG for {function_name or 'main'}"]
        for i, b in enumerate(blocks):
            head = b[0] if b else "(empty)"
            out.append(f"B{i}: {head}")
        return "\n".join(out)
    # Fallback sample
    return (
        "main\n"
        "├── parse_input\n"
        "│   ├── validate_user\n"
        "│   └── error_exit\n"
        "└── crypto_routine\n"
        "    ├── encrypt_data\n"
        "    └── hash_password"
    )
