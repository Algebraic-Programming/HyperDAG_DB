import os
import sys
import io
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import functools
from typing import Tuple, List


def _read_matrix_market(filename: str) -> nx.DiGraph:
    G = nx.DiGraph()
    # Open file using utf-8 encoding
    with io.open(filename, "r", encoding="utf-8") as fp:
        # Read the header
        header = fp.readline()
        # Check if header is valid
        if not header.startswith("%%MatrixMarket matrix coordinate"):
            raise ValueError(f"Invalid header: {header}")
        # Read the comment line(s)
        current_line = fp.readline()
        while current_line.startswith("%"):
            current_line = fp.readline()
            pass
        # Read the information of the matrix: nrows, ncols, nnz
        nnz = int(current_line.split()[2])
        print(f"Reading {nnz} edges...")
        # Read the data
        for _ in range(nnz):
            current_line = fp.readline()
            row = int(current_line.split()[0])
            col = int(current_line.split()[1])
            val = float(current_line.split()[2])
            G.add_edge(row, col, weight=val)
        return G

def _find_sink_nodes(G: nx.DiGraph) -> list:
    return [node for node in G.nodes if G.out_degree(node) == 0]

def _font_weight_sink_nodes(G: nx.DiGraph, sink_nodes: list, not_sink="normal", sink="bold") -> List[str]:
    # Weight sink nodes
    weights = [ not_sink if sink_nodes.count(node) == 0 else sink for node in G.nodes ]
    return weights

def _node_color_sink_nodes(G: nx.DiGraph, sink_nodes: list, not_sink="k", sink="r") -> List[str]:
    # Colorise sink nodes
    colours = [ not_sink if sink_nodes.count(node) == 0 else sink for node in G.nodes ]
    return colours

def _lerp(minimum, maximum, value):
    return minimum + (maximum - minimum) * value

@functools.lru_cache
def _get_weights(G: nx.DiGraph) -> Tuple[float, float, List[float]]:
    weights = [ edge[2]["weight"] for edge in G.edges(data=True) ]
    return (
        min(weights),
        max(weights),
        weights
    )

def _arrowsize_by_weight(G: nx.DiGraph, min_size=10, max_size=20) -> List[float]:
    min_w, max_w, edges_weights = _get_weights(G)
    if min_w == max_w:
        return [ min_size for _ in edges_weights ]
    return [ _lerp(min_size, max_size, (w - min_w) / (max_w - min_w)) for w in edges_weights ]

def _edgewidth_by_weight(G: nx.DiGraph, min_width=0.5, max_width=3) -> list:
    min_w, max_w, edges_weights = _get_weights(G)
    if min_w == max_w:
        return [ min_width for _ in edges_weights ]
    return [ _lerp(min_width, max_width, (w - min_w) / (max_w - min_w)) for w in edges_weights ]

def command(filename) -> None:
    G = _read_matrix_market(filename)
    sink_nodes = _find_sink_nodes(G)
    nx.draw(
        G,
       # pos=nx.kamada_kawai_layout(G),
        with_labels=True,
        arrows=True,
        node_size=200,
        font_size=8,
        arrowsize=_arrowsize_by_weight(G),
        width=_edgewidth_by_weight(G),
        font_color="white",
        edge_color="gray",
        node_color=_node_color_sink_nodes(G, sink_nodes)
    )
    # Save to file named filename + '.png' (1920x1080)
    plt.savefig(filename + ".png", dpi=312)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualise_dag.py <filename>")
        sys.exit(1)
    command(sys.argv[1])
