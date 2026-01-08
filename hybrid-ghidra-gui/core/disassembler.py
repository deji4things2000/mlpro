from __future__ import annotations

from typing import List, Tuple
from pathlib import Path


def _read_bytes(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as f:
        f.seek(offset)
        return f.read(size)


def _disassemble_x64(code_bytes: bytes, start_addr: int) -> List[str]:
    try:
        from capstone import Cs, CS_ARCH_X86, CS_MODE_64
        md = Cs(CS_ARCH_X86, CS_MODE_64)
        lines = []
        for ins in md.disasm(code_bytes, start_addr):
            lines.append(f"{ins.address:016X}  {ins.mnemonic} {ins.op_str}".rstrip())
        return lines
    except Exception:
        return []


def _pe_preview(path: Path) -> Tuple[str, List[Tuple[str, str, str]]]:
    try:
        import pefile
        pe = pefile.PE(str(path), fast_load=True)
        pe.parse_data_directories()
        entry_rva = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        image_base = pe.OPTIONAL_HEADER.ImageBase
        entry_off = pe.get_offset_from_rva(entry_rva)
        code_bytes = _read_bytes(path, entry_off, 512)
        lines = _disassemble_x64(code_bytes, image_base + entry_rva)
        asm = "\n".join(lines) if lines else "(No disassembly produced)"
        funcs = [("entry_point", f"0x{image_base + entry_rva:X}", "Unknown")]
        return asm, funcs
    except Exception:
        return "(PE preview failed)", [("entry_point", "0x0", "Unknown")]


def detect_format(path: str) -> str:
    p = Path(path)
    try:
        with p.open("rb") as f:
            magic = f.read(4)
        if magic.startswith(b"MZ"):
            return "pe"
        if magic in (b"\x7FELF",):
            return "elf"
        if magic in (b"\xCF\xFA\xED\xFE", b"\xCE\xFA\xED\xFE"):
            return "mach-o"
        return "unknown"
    except Exception:
        return "unknown"


def get_preview(path: str) -> Tuple[str, List[Tuple[str, str, str]]]:
    fmt = detect_format(path)
    p = Path(path)
    if fmt == "pe":
        return _pe_preview(p)
    # Fallback: naive x86_64 disassembly of first bytes
    try:
        code_bytes = _read_bytes(p, 0, 512)
        lines = _disassemble_x64(code_bytes, 0x0)
        asm = "\n".join(lines) if lines else "(No disassembly produced)"
        return asm, [("entry_point", "0x0", "Unknown")]
    except Exception:
        return "(Preview failed)", [("entry_point", "0x0", "Unknown")]
