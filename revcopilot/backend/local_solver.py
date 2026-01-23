#!/usr/bin/env python3
"""
RevCopilot Local Solver
Solves classic crackmes using angr + Z3 + pattern detection.
"""

import angr
import claripy
import capstone
import binascii
import logging
from typing import Optional, Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class LocalCrackmeSolver:
    def __init__(self, binary_path: str):
        self.binary_path = binary_path
        self.proj = angr.Project(binary_path, auto_load_libs=False)
        self.md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        self.results = {}
        
    def find_string_refs(self, target_str: bytes) -> list:
        """Find addresses referencing a given string."""
        refs = []
        try:
            for section in self.proj.loader.main_object.sections:
                if section.is_executable:
                    data = section.data()
                    for i in range(len(data) - len(target_str) + 1):
                        if data[i:i+len(target_str)] == target_str:
                            refs.append(section.vaddr + i)
        except Exception as e:
            log.warning(f"String scan failed: {e}")
        return refs
    
    def identify_check_functions(self) -> Dict[str, Any]:
        """Locate success/failure strings and their functions."""
        checks = {}
        # Common crackme strings
        strings = {
            "success": [b"solved", b"correct", b"success", b"win"],
            "failure": [b"incorrect", b"wrong", b"fail", b"nope"],
            "prompt": [b"enter", b"password", b"key", b"flag"]
        }
        
        for category, str_list in strings.items():
            for s in str_list:
                refs = self.find_string_refs(s)
                if refs:
                    checks[category] = {"string": s, "addrs": refs}
                    break
        return checks
    
    def symbolic_solve(self, arg_sizes: Tuple[int, int] = (16, 16)):
        """Symbolic execution to find inputs."""
        log.info("Starting symbolic execution...")
        
        # Create symbolic argv
        arg1 = claripy.BVS('arg1', 8 * arg_sizes[0])
        arg2 = claripy.BVS('arg2', 8 * arg_sizes[1])
        argv = [self.binary_path, arg1, arg2]
        
        # Initial state
        state = self.proj.factory.entry_state(args=argv)
        
        # Constrain to printable ASCII
        for byte in arg1.chop(8):
            state.solver.add(byte >= 0x20, byte <= 0x7e)
        for byte in arg2.chop(8):
            state.solver.add(byte >= 0x20, byte <= 0x7e)
        
        # Find success addresses
        checks = self.identify_check_functions()
        success_addrs = checks.get("success", {}).get("addrs", [])
        fail_addrs = checks.get("failure", {}).get("addrs", [])
        
        if not success_addrs:
            log.warning("No success string found, using exploration.")
            # Explore generically
            simgr = self.proj.factory.simulation_manager(state)
            simgr.run()
            if simgr.deadended:
                # Try to extract from deadended states
                for s in simgr.deadended:
                    # Heuristic: look for states that printed something
                    pass
            return None
            
        # Target success address
        target = success_addrs[0]
        log.info(f"Target success address: 0x{target:x}")
        
        simgr = self.proj.factory.simulation_manager(state)
        simgr.explore(find=target, avoid=fail_addrs)
        
        if simgr.found:
            sol_state = simgr.found[0]
            arg1_val = sol_state.solver.eval(arg1, cast_to=bytes).decode('ascii', errors='ignore')
            arg2_val = sol_state.solver.eval(arg2, cast_to=bytes).decode('ascii', errors='ignore')
            log.info(f"Found solution: arg1='{arg1_val}', arg2='{arg2_val}'")
            return (arg1_val, arg2_val)
        else:
            log.warning("Symbolic execution did not find solution.")
            return None
    
    def pattern_detect_transform(self) -> Optional[Dict[str, Any]]:
        """Static pattern detection for common transforms."""
        transforms = []
        # Scan for XOR patterns, ROL, ADD, SUB, etc.
        for section in self.proj.loader.main_object.sections:
            if section.is_executable:
                code = section.data()
                for insn in self.md.disasm(code, section.vaddr):
                    # Look for XOR reg, imm
                    if 'xor' in insn.mnemonic and insn.op_str:
                        if '0x' in insn.op_str:
                            transforms.append({"type": "xor", "insn": f"{insn.mnemonic} {insn.op_str}"})
                    # Look for rotation
                    if 'rol' in insn.mnemonic or 'ror' in insn.mnemonic:
                        transforms.append({"type": "rotate", "insn": f"{insn.mnemonic} {insn.op_str}"})
        return transforms if transforms else None
    
    def solve(self) -> Dict[str, Any]:
        """Main solving pipeline."""
        log.info(f"Analyzing {self.binary_path}")
        
        # 1. Detect transformations
        transforms = self.pattern_detect_transform()
        
        # 2. Symbolic solve
        solution = self.symbolic_solve()
        
        # 3. Build result
        self.results = {
            "transforms_detected": transforms,
            "solution": solution,
            "checks": self.identify_check_functions()
        }
        return self.results
    
    def print_report(self):
        """Print formatted results."""
        print("\n" + "="*60)
        print("RevCopilot - Analysis Report")
        print("="*60)
        
        if self.results.get("transforms_detected"):
            print("\n[+] Detected Transformations:")
            for t in self.results["transforms_detected"]:
                print(f"    • {t['type']}: {t['insn']}")
        
        if self.results.get("checks"):
            print("\n[+] Located Strings:")
            for cat, info in self.results["checks"].items():
                print(f"    • {cat}: {info['string']} at {[hex(a) for a in info['addrs']]}")
        
        if self.results.get("solution"):
            arg1, arg2 = self.results["solution"]
            print("\n[+] SOLUTION FOUND!")
            print(f"    argv[1] = {arg1}")
            print(f"    argv[2] = {arg2}")
            print("\n[+] Command:")
            print(f"    ./{self.binary_path} '{arg1}' '{arg2}'")
        else:
            print("\n[-] No solution found automatically.")
            print("    Try AI‑assisted mode or manual analysis.")

# CLI interface
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <binary>")
        sys.exit(1)
    
    solver = LocalCrackmeSolver(sys.argv[1])
    results = solver.solve()
    solver.print_report()