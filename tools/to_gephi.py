import os
import sys
import io
import networkx as nx

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
        print(f"Reading {nnz} edges from {filename} ...")
        # Read the data
        for _ in range(nnz):
            current_line = fp.readline()
            row = int(current_line.split()[0])
            col = int(current_line.split()[1])
            val = float(current_line.split()[2])
            G.add_edge(row, col, weight=val)
        return G


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualise_dag.py <filename>")
        sys.exit(1)
    filename = sys.argv[1]
    G = _read_matrix_market(filename)
    filename_noext = os.path.splitext(filename)[0]
    nx.write_gexf(G, f"{filename_noext}.gexf")
