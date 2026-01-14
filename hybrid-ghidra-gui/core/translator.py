def to_python(c_code: str) -> str:
    """Very rough translation of C-like pseudocode to Python.
    Strips types/braces/semicolons and converts basic control flow.
    This is a heuristic, not a recompilable translation.
    """
    lines = []
    indent = 0
    for raw in c_code.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        # Handle block end
        if line.startswith('}'):
            indent = max(0, indent - 1)
            continue
        # Handle block start
        if line.endswith('{'):
            # Convert control flow headers
            header = line[:-1].strip()
            header_py = header
            header_py = header_py.replace('else if', 'elif')
            header_py = header_py.replace('if (', 'if ').replace(') {', '')
            header_py = header_py.replace('while (', 'while ').replace(') {', '')
            header_py = header_py.replace('for (', '# for ').replace(') {', '')
            lines.append(('    ' * indent) + header_py + ':')
            indent += 1
            continue
        # Remove trailing semicolons and types
        py = line.rstrip(';')
        for t in ('int ', 'char ', 'void ', 'long ', 'short ', 'float ', 'double '):
            py = py.replace(t, '')
        # Replace common funcs and operators
        py = py.replace('&&', 'and').replace('||', 'or').replace('!', 'not ')
        # Basic return/printf conversions
        if py.startswith('return '):
            py = 'return ' + py[len('return '):]
        py = py.replace('printf(', 'print(')
        # Assignments stay as-is; pointers/arrays unresolved
        lines.append(('    ' * indent) + py)
    header = (
        "# NOTE: Heuristic translation from C-like pseudocode to Python\n"
        "# Result is for readability, not for execution\n"
    )
    return header + '\n'.join(lines)


def to_cpp(c_code: str) -> str:
    """Pass-through normalization for C++: returns the input with a header.
    For now, treat Ghidra/RetDec C-like pseudocode as C++; users can refine types.
    """
    header = (
        "// NOTE: Pseudocode normalized for C++; manual type refinement may be required\n"
    )
    return header + c_code
