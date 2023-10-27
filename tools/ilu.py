import sys
import os
import numpy as np
import scipy.sparse as sp
import scipy.io as sp_io


def _assert_file_exists(filename: str) -> None:
    if not os.path.isfile(filename):
        print(f"Error: {filename} is not an existing file", file=sys.stderr)
        sys.exit(1)


def ilu(A, threshold: float = 1e-4):
    invA = sp.linalg.spilu(A, drop_tol=threshold)
    return sp.csr_matrix(invA.L), sp.csr_matrix(invA.U)


def read_mtx(filename: str) -> sp.csc_matrix:
    with open(filename, "r") as f:
        current_line = f.readline()

        # Read header
        header = current_line.strip().split()
        assert header[0] == "%%MatrixMarket"
        assert header[1] == "matrix"
        assert header[2] == "coordinate"
        assert header[3] == "real"
        if len(header) == 5:
            assert header[4] == "general"
        else:
            assert header[4] == "symmetric"
            assert header[5] == "general"

        current_line = f.readline()
        while current_line.startswith("%"):
            current_line = f.readline()

        # Read matrix size
        size = current_line.strip().split()
        nrows = int(size[0])
        ncols = int(size[1])
        nnz = int(size[2])

        # Skip comments
        current_line = f.readline()
        while current_line.startswith("%"):
            current_line = f.readline()

        # Read matrix entries
        entries = np.zeros((nnz, 3))
        for i in range(nnz):
            while current_line.startswith("%") == '%':
                current_line = f.readline()
            row, col, val = current_line.strip().split()
            row = int(row)-1
            col = int(col)-1
            val = float(val)
            # print(f"({row}, {col}) = {val}")
            entries[i] = [row, col, val]
            assert entries[i, 0] <= nrows
            assert entries[i, 1] <= ncols

            current_line = f.readline()

        # Create sparse matrix
        matrix = sp.csc_matrix(
            (entries[:, 2], (entries[:, 0], entries[:, 1])), shape=(nrows, ncols))
        return matrix


def main(intput_filename: str, output_filename_L: str, output_filename_U: str):
    A = read_mtx(intput_filename)
    L, U = ilu(A)
    # Save L and U to files (MatrixMarket format)
    sp_io.mmwrite(output_filename_L, L)
    sp_io.mmwrite(output_filename_U, U)
    pass


if __name__ == '__main__':
    # Check usage
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print(
            f"Usage: {sys.argv[0]} <input_filename> [output_filename_L] [output_filename_U]", file=sys.stderr)
        sys.exit(1)

    # Check if input_filename is a correct and an existing file
    input_filename = sys.argv[1]
    _assert_file_exists(input_filename)

    # Attribute a default output L filename if none is given
    output_filename_L = ""
    if len(sys.argv) < 3:
        output_filename_L = os.path.splitext(input_filename)[0] + "_L.mtx"
    else:
        output_filename_L = sys.argv[2]
        _assert_file_exists(output_filename_L)
    print(f"Output file for L: {output_filename_L}")

    # Attribute a default output U filename if none is given
    output_filename_U = ""
    if len(sys.argv) < 4:
        output_filename_U = os.path.splitext(input_filename)[0] + "_U.mtx"
    else:
        output_filename_U = sys.argv[3]
        _assert_file_exists(output_filename_U)
    print(f"Output file for U: {output_filename_U}")

    # Call the main function
    main(input_filename, output_filename_L, output_filename_U)
