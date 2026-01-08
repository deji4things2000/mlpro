from utils.file_utils import read_file_bytes


class GhidraBridge:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.connected = False

    def connect(self):
        # Placeholder: in a real setup, use ghidra_bridge to connect to Ghidra
        self.connected = True
        self.logger.info("Ghidra bridge (stub) connected.")

    def disassemble(self, path: str):
        # Minimal stub: produce pseudo-disassembly from bytes
        data = read_file_bytes(path)
        lines = []
        for i in range(0, min(len(data), 256), 8):
            chunk = data[i:i+8]
            addr = f"{i:08x}:"
            hexbytes = " ".join(f"{b:02x}" for b in chunk)
            lines.append(f"{addr} db {hexbytes} ; bytes")
        if not lines:
            lines = ["00000000: ret ; empty"]
        return lines
