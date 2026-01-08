import networkx as nx


def build_cfg(disassembly_lines):
    G = nx.DiGraph()
    prev = None
    for idx, line in enumerate(disassembly_lines or []):
        node = idx
        G.add_node(node, label=line)
        if prev is not None:
            G.add_edge(prev, node)
        prev = node
    return G
