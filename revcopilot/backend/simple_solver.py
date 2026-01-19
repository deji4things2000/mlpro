#!/usr/bin/env python3
"""
Simple solver for medium.bin using known transformation.
"""

def solve_part1():
    """Solve using the transformation we reversed earlier."""
    target = bytes([0xa5, 0xa5, 0xc5, 0x04, 0xe4, 0xa5, 0x35, 0x04,
                    0x75, 0xa5, 0x44, 0x75, 0x14, 0xc4, 0xd4, 0x24])
    key = 0x05
    
    def ror4(b):
        return ((b >> 4) | (b << 4)) & 0xFF
    
    # XOR swap (mirror bytes)
    swapped = bytearray(target)
    left, right = 0, 15
    while left < right:
        swapped[left] ^= swapped[right]
        swapped[right] ^= swapped[left]
        swapped[left] ^= swapped[right]
        left += 1
        right -= 1
    
    # Reverse ROL4 and XOR
    result = bytearray()
    for b in swapped:
        b = ror4(b)  # undo ROL4
        b ^= key     # undo XOR
        result.append(b)
    
    return result.decode('ascii')

def solve_part2():
    """Part 2 key from our GDB analysis."""
    return "TR_C31NG_KEY_2__"

if __name__ == "__main__":
    arg1 = solve_part1()
    arg2 = solve_part2()
    print(f"Solution for medium.bin:")
    print(f"  argv[1] = {arg1}")
    print(f"  argv[2] = {arg2}")
    print(f"\nRun: ./medium.bin '{arg1}' '{arg2}'")