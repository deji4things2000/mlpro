from pathlib import Path

# Simple x86-ish byte pattern: prologue, call, jmp, ret
BYTES = bytes([
    0x55,              # push rbp
    0x48, 0x89, 0xE5,  # mov rbp, rsp
    0xE8, 0x00, 0x00, 0x00, 0x00,  # call rel32 (dummy)
    0xE9, 0x00, 0x00, 0x00, 0x00,  # jmp rel32 (dummy)
    0x90,              # nop
    0xC3               # ret
])

out_dir = Path(__file__).resolve().parents[1] / "assets"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "sample.bin"
out_file.write_bytes(BYTES)
print(f"Wrote: {out_file} ({len(BYTES)} bytes)")
