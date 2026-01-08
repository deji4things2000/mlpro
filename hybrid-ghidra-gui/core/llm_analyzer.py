def analyze(disassembly_lines, cfg=None):
    if not disassembly_lines:
        return "No disassembly available."

    total = len(disassembly_lines)
    suspicious = [l for l in disassembly_lines if "jmp" in l or "call" in l]
    report = []
    report.append(f"Instructions analyzed: {total}")
    report.append(f"Potential control transfers: {len(suspicious)}")
    if cfg is not None:
        try:
            nodes = len(cfg.nodes)
            edges = len(cfg.edges)
            report.append(f"CFG nodes: {nodes}, edges: {edges}")
        except Exception:
            pass
    report.append("Heuristic analysis only. Configure LLM for deeper insights.")
    return "\n".join(report)
