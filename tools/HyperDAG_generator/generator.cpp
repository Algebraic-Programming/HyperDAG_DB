/*
Copyright 2022 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Pal Andras Papp
*/

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <execution>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;
using num_t = size_t;

bool DebugMode = false;

static const map<string, string> modes_usage = {
    {"help", "<mode>"},
    {"ER", "Undescribed usage"},
    {"fixedIn", "Undescribed usage"},
    {"expectedIn", "Undescribed usage"},
    {"SpMV",
     "{-N <n> -nonzeroProb <nonzeroProb>} | {-inputs <matrix_filename>}"},
    {"SpMVExp",
     "-K <k> {-N <n> -nonzeroProb <nonzeroProb>} | {-inputs "
     "<matrix_filename>}"},
    {"LLtSolver",
     "{-N <n> -nonzeroProb <nonzeroProb>} | {-inputs <matrix_filename>}"},
    {"LLtSolverExp",
     "-K <k> {-N <n> -nonzeroProb <nonzeroProb>} | {-inputs "
     "<matrix_filename>}"},
    {"ALLtSolver",
     "{-N <n> -nonzeroProb <nonzeroProb>} | {-inputs <matrix_A_filename> "
     "<matrix_L_filename>}"},
    {"ALLtSolverExp",
     "-K <k> -nonzeroProb <nonzeroProb>} | {-inputs <matrix_A_filename> "
     "<matrix_L_filename>}"},
    {"LUSolver",
     "{-N <n> -nonzeroProb <nonzeroProb>} | {-inputs <matrix_L_filename> "
     "<matrix_U_filename>}"},
    {"LUSolverExp",
     "-K <k> {-N <n> -nonzeroProb <nonzeroProb>} | {-inputs "
     "<matrix_L_filename> <matrix_U_filename>}"},
    {"ALUSolver",
     "{-N <n> -nonzeroProb <nonzeroProb>} | {-inputs <matrix_A_filename> "
     "<matrix_L_filename> <matrix_U_filename>}"},
    {"ALUSolverExp",
     "-K <k> {-N <n> -nonzeroProb <nonzeroProb>} | {-inputs "
     "<matrix_A_filename> <matrix_L_filename> <matrix_U_filename>}"},
    {"kNN", "Undescribed usage"},
    {"CG", "Undescribed usage"}};

static string str_tolower(const string& s_init) {
    string s = s_init;
    std::transform(s_init.cbegin(), s_init.cend(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); }  // correct
    );
    return s;
}

static vector<string>& getModes() {
    static vector<string> modes;
    if (modes.empty()) {
        modes.reserve(modes_usage.size());
        for (const auto& entry : modes_usage) {
            modes.push_back(str_tolower(entry.first));
        }
    }
    return modes;
}

// AUXILIARY FUNCTIONS

// unbiased random num_t generator
num_t randInt(num_t lim) {
    num_t rnd = rand();
    while (rnd >= RAND_MAX - RAND_MAX % lim) rnd = rand();
    return rnd % lim;
}

// check if vector contains a specific element
bool contains(const vector<string>& vec, const string& s) {
    return std::find(vec.begin(), vec.end(), s) != vec.end();
}

// MAIN DATA STRUCTURES

struct DAG {
   private:
    num_t n;
    unordered_map<num_t, vector<num_t>> _In, _Out;
    vector<string> descriptions;

    vector<num_t>& In(const num_t i) {
        // Return a reference to the vector of predecessors of node i (only if
        // it exists)
        _In.try_emplace(i, vector<num_t>());
        return _In[i];
    }
    vector<num_t>& Out(const num_t i) {
        // Return a reference to the vector of successors of node i (only if it
        // exists)
        _Out.try_emplace(i, vector<num_t>());
        return _Out[i];
    }

    const vector<num_t>& In(const num_t i) const {
        static const vector<num_t> empty(0);
        return _In.find(i) != _In.cend() ? _In.at(i) : empty;
    }

    const vector<num_t>& Out(const num_t i) const {
        static const vector<num_t> empty(0);
        return _Out.find(i) != _Out.cend() ? _Out.at(i) : empty;
    }

    num_t getNnz() const {
        return accumulate(_Out.cbegin(), _Out.cend(), static_cast<num_t>(0),
                          [](num_t sum, const pair<num_t, vector<num_t>>& p) {
                              return sum + p.second.size();
                          });
    }

   public:
    DAG(num_t N = 0) : n(N), _In(N), _Out(N) {}

    num_t size() const { return n; }

    void resize(num_t N) {
        cout << "Resizing DAG from " << n << " to " << N << endl;
        if (N <= n) return;
        n = N;
    }

    void addDescriptionLine(const string& line) {
        // Split the line using the character '\n'
        stringstream ss(line);
        string token;
        while (getline(ss, token, '\n')) descriptions.push_back(token);
    }

    void addEdge(num_t v1, num_t v2, const string& description = "",
                 bool noPrint = false) {
        if (DebugMode && !noPrint) {
            cout << "New edge ( " << setw(3) << setfill('0') << v1 << " -> "
                 << setw(3) << setfill('0') << v2 << " ) ";
            cout << "{ note: " << description << " }\n" << flush;
        }
        if (v1 == v2) {
            cerr << "Self-loop edge addition error: " << v1 << " -> " << v2
                 << endl;
            abort();
        }

        if (v1 >= n || v2 >= n) {
            cerr << "DAG edge addition error: node index out of range." << endl;
            cerr << "v1: " << v1 << ", v2: " << v2 << endl;
            cerr << "n: " << n << endl;
            abort();
        }

        In(v2).push_back(v1);
        Out(v1).push_back(v2);
    }

    void printDAG(const string& filename) const {
        cout << "Printing DAG to file: " << filename << endl
             << "  # Header: (n=" << n << ", m=" << n << ", nnz=" << getNnz()
             << ")" << endl;

        ofstream outfile(filename, ios::out);
        outfile << "%%MatrixMarket matrix coordinate real general\n";
        outfile << "%\n";
        for (const auto& each : descriptions) outfile << "% " << each << "\n";
        outfile << "%\n";
        // Print MM header
        outfile << n << " " << n << " " << getNnz() << "\n";
        // Print edges (directed)
        for (num_t i = 0; i < n; ++i) {
            for (num_t j = 0; j < Out(i).size(); ++j) {
                outfile << i << " " << Out(i)[j] << " "
                        << "1"
                        << "\n";
            }
        }
    }

    // Prints the hyperDAG corresponding to the DAG into a file
    void printHyperDAG(const string& filename) const {
        cout << "Printing HyperDAG to file: " << filename << endl;
        num_t sinks = 0, pins = 0;
        for (num_t i = 0; i < n; ++i) {
            if (Out(i).empty()) {
                ++sinks;
            } else {
                pins += 1 + Out(i).size();
            }
        }

        ofstream outfile;
        outfile.open(filename);
        for (const auto& each : descriptions) outfile << "% " << each << "\n";
        outfile << "%\n";
        outfile << "% Hyperedges: " << n - sinks << "\n";
        outfile << "% Nodes: " << n << "\n";
        outfile << "% Pins: " << pins << "\n";
        outfile << n - sinks << " " << n << " " << pins << "\n";

        // Print communication weights of hyperedges - uniformly 1
        outfile << "% Hyperedges ( id, communication cost ):\n";
        num_t edgeIndex = 0;
        for (num_t i = 0; i < n; ++i) {
            if (Out(i).empty()) continue;

            outfile << edgeIndex << " " << 1 << "\n";
            ++edgeIndex;
        }

        // Print work weights of nodes - this is indegree-1
        outfile << "% Nodes ( id, work cost ):\n";
        for (num_t i = 0; i < n; ++i) {
            if (Out(i).empty() && In(i).empty()) continue;

            outfile << i << " " << (In(i).size() == 0 ? 0 : In(i).size() - 1)
                    << "\n";
        }

        // Print all pins
        outfile << "% Pins ( hyperdedge.id, node.id ):\n";
        edgeIndex = 0;
        for (num_t i = 0; i < n; ++i) {
            if (Out(i).empty()) {
                continue;
            }

            outfile << edgeIndex << " " << i << "\n";
            for (num_t j = 0; j < Out(i).size(); ++j)
                outfile << edgeIndex << " " << Out(i)[j] << "\n";

            ++edgeIndex;
        }
        outfile.close();
    }

    // Checks for each node whether it is connected (in an undirected sense) to
    // the set of source nodes (using a BFS) (if the onlyBack flag is set, then
    // we only search for the predecessors of the given nodes)
    vector<bool> isReachable(const vector<num_t>& sources,
                             bool onlyBack = false) const {
        vector<bool> visited(n, false);
        list<num_t> next;

        // Mark the input nodes as reachable
        for (num_t i = 0; i < sources.size(); ++i) {
            visited[sources[i]] = true;
            next.push_back(sources[i]);
        }

        // Execute BFS
        while (!next.empty()) {
            num_t node = next.front();
            next.pop_front();

            for (num_t i = 0; i < In(node).size(); ++i)
                if (!visited[In(node)[i]]) {
                    next.push_back(In(node)[i]);
                    visited[In(node)[i]] = true;
                }

            if (onlyBack) continue;

            for (num_t i = 0; i < Out(node).size(); ++i)
                if (!visited[Out(node)[i]]) {
                    next.push_back(Out(node)[i]);
                    visited[Out(node)[i]] = true;
                }
        }

        return visited;
    }

    // Checks if the DAG is a single connected component
    bool isConnected() const {
        vector<num_t> sources(1, 0);
        vector<bool> reachable = isReachable(sources);

        for (num_t i = 0; i < n; ++i)
            if (!reachable[i]) return false;

        return true;
    }

    // Creates a smaller, 'cleaned' DAG, consisting of only the specified nodes
    DAG keepGivenNodes(const vector<bool>& keepNode) {
        num_t NrOfNodes = 0;
        vector<num_t> newIdx(n);
        for (num_t i = 0; i < n; ++i)
            if (keepNode[i]) {
                newIdx[i] = i;  // NrOfNodes;
                ++NrOfNodes;
            }

        DAG cleaned(n);
        for (num_t i = 0; i < descriptions.size(); ++i)
            cleaned.addDescriptionLine(descriptions[i]);

        for (num_t i = 0; i < n; ++i)
            if (keepNode[i])
                for (num_t j = 0; j < Out(i).size(); ++j)
                    if (keepNode[Out(i)[j]])
                        cleaned.addEdge(newIdx[i], newIdx[Out(i)[j]], "", true);

        if (DebugMode) {
            cout << "Only the following nodes are kept: ";
            for (num_t i = 0; i < keepNode.size(); ++i)
                if (keepNode[i]) cout << i << " ";
            cout << endl;
        }

        return (cleaned);
    }

    void printConnected() const {
        if (isConnected())
            cout << "The DAG is connected.\n";
        else
            cout << "The DAG is NOT connected.\n";
    }
};

struct IMatrix {
   private:
    const num_t _m, _n;
    string _label = "";
    string _desc = "";

   protected:
    IMatrix(num_t M, num_t N, const string& label = "")
        : _m(M), _n(N), _label(label) {
        std::cout << "IMatrix(" << M << ", " << N << ", " << label << ")\n";
    }

    IMatrix& operator=(const IMatrix& other) {
        assert(_m == other._m && _n == other._n);
        _label = other._label;
        _desc = other._desc;
        return *this;
    }

   public:
    num_t nrows() const { return _m; }
    num_t ncols() const { return _n; }
    num_t area() const { return nrows() * ncols(); }
    virtual num_t nnz() const = 0;

    void addDescription(const string& desc) { _desc += desc + "\n"; }
    std::string getDescription() const { return _desc; }
    std::string getLabel() const { return _label; }

    virtual bool at(num_t i, num_t j) const = 0;
    virtual void set(num_t i, num_t j, bool value) = 0;

    void setMainDiagonal() {
        for (num_t i = 0; i < nrows(); ++i) {
            set(i, i, true);
        }
    }

    /**
     * Randomized matrix where each cell is nonzero independently with a fixed
     * probability (if the mainDiag flag is set to true, then the main diagonal
     * is always set to 1)
     *
     * @param nonzero Probability for having a nonzero in any cell of matrix A
     */
    void randomize(double nonzero) {
        for (num_t i = 0; i < nrows(); ++i) {
            for (num_t j = 0; j < ncols(); ++j) {
                double rnd = (double)rand() / (double)RAND_MAX;
                set(i, j, rnd < nonzero);
            }
        }

        addDescription(
            "The probability for having a nonzero in any cell of matrix " +
            getLabel() + " is " + to_string(nonzero) + ".");

        if (DebugMode) {
            cout << "Nonzero count: " << nnz() << " out of "
                 << nrows() * ncols() << ", which is "
                 << (double)nnz() / ((double)nrows() * (double)ncols())
                 << " instead of " << nonzero << endl;
        }
    }

    void print(const string& label = "", std::ostream& os = std::cout) const {
        os << "Matrix " << label << ": " << nrows() << "x" << ncols() << ", "
           << nnz() << " nonzeros\n";
        os << "[\n";
        if (nrows() > 10 || ncols() > 01) {
            os << "  ... (matrix too large to print)\n";
            os << "]\n";
            return;
        }
        for (num_t i = 0; i < nrows(); ++i) {
            os << "  ";
            for (num_t j = 0; j < ncols(); ++j) {
                if (at(i, j)) {
                    os << "X";
                } else {
                    os << "_";
                }
                os << " ";
            }
            os << "\n";
        }
        os << "]\n";
    }

    virtual ~IMatrix() {}

    template <typename MatrixType>
    static MatrixType readFromFile(const string& filename,
                                   bool IndexedFromOne = true) {
        ifstream infile(filename);
        if (!infile.is_open()) {
            throw "Unable to find/open input matrix file.\n";
        }

        string line;
        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%') getline(infile, line);

        num_t M, N, NNZ;
        sscanf(line.c_str(), "%ld %ld %ld", &N, &M, &NNZ);

        if (NNZ == 0) {
            throw runtime_error(
                "Incorrect input file format: only non-empty matrices are "
                "accepted!\n");
        }

        MatrixType A(M, N, "Matrix from file <" + filename + ">");
        // read nonzeros
        num_t indexOffset = IndexedFromOne ? 1 : 0;
        for (num_t i = 0; i < NNZ; ++i) {
            if (infile.eof()) {
                throw runtime_error(
                    "Incorrect input file format (file terminated too "
                    "early).\n");
            }

            num_t x, y;
            getline(infile, line);
            while (!infile.eof() && line.at(0) == '%') getline(infile, line);

            sscanf(line.c_str(), "%ld %ld", &x, &y);

            if (x < indexOffset || y < indexOffset || x >= N + indexOffset ||
                y >= N + indexOffset) {
                throw runtime_error(
                    "Incorrect input file format (index out of range).\n");
            }

            A.set(x - indexOffset, y - indexOffset, true);
        }

        A.addDescription("Matrix A is read from input file " + filename + ".");
        infile.close();

        return A;
    }

    template <typename MatrixType>
    static MatrixType Identity(num_t M, num_t N) {
        MatrixType I(M, N);
        I.setMainDiagonal();
        return I;
    }
};

struct SquareMatrix : public IMatrix {
   private:
    num_t _nnz = 0;
    vector<vector<bool>> _cells;

   public:
    SquareMatrix(num_t N = 0, const string& label = "") : IMatrix(N, N, label) {
        _cells.resize(N, vector<bool>(N, false));

        addDescription("Matrix A is a square-matrix of size " + to_string(N) +
                       "x" + to_string(N) + ".");
    }

    SquareMatrix(num_t _, num_t N, const string& label)
        : SquareMatrix(N, label) {}

    SquareMatrix(const SquareMatrix& other) : IMatrix(other) {
        _nnz = other._nnz;
        _cells = other._cells;
    }

    SquareMatrix& operator=(const SquareMatrix& other) {
        IMatrix::operator=(other);
        _nnz = other._nnz;
        _cells = std::move(other._cells);
        return *this;
    }

    num_t nnz() const override { return _nnz; }

    bool at(num_t i, num_t j) const override { return _cells[i][j]; }

    void set(num_t i, num_t j, bool value) override {
        if (at(i, j) == value) return;
        _cells[i][j] = value;

        if (value)
            _nnz++;
        else if (_nnz > 0)
            _nnz--;
    }
};

struct LowerTriangularSquareMatrix : public SquareMatrix {
   public:
    LowerTriangularSquareMatrix(num_t N = 0, const string& label = "")
        : SquareMatrix(N, label) {
        addDescription("Matrix A is a lower triangular square-matrix of size " +
                       to_string(N) + "x" + to_string(N) + ".");
    }

    LowerTriangularSquareMatrix(num_t _, num_t N, const string& label)
        : SquareMatrix(N, label) {}

    LowerTriangularSquareMatrix(const LowerTriangularSquareMatrix& other)
        : SquareMatrix(other) {}

    LowerTriangularSquareMatrix& operator=(
        const LowerTriangularSquareMatrix& other) {
        SquareMatrix::operator=(other);
        return *this;
    }

    bool at(num_t i, num_t j) const override {
        if (i < j) return false;
        return SquareMatrix::at(i, j);
    }

    void set(num_t i, num_t j, bool value) override {
        if (i < j) return;
        SquareMatrix::set(i, j, value);
    }
};

struct UpperTriangularSquareMatrix : public SquareMatrix {
   public:
    UpperTriangularSquareMatrix(num_t N = 0, const string& label = "")
        : SquareMatrix(N, label) {
        addDescription("Matrix A is a upper triangular square-matrix of size " +
                       to_string(N) + "x" + to_string(N) + ".");
    }

    UpperTriangularSquareMatrix(num_t _, num_t N, const string& label)
        : SquareMatrix(N, label) {}

    UpperTriangularSquareMatrix(const UpperTriangularSquareMatrix& other)
        : SquareMatrix(other) {}

    UpperTriangularSquareMatrix& operator=(
        const UpperTriangularSquareMatrix& other) {
        SquareMatrix::operator=(other);
        return *this;
    }

    bool at(num_t i, num_t j) const override {
        if (i > j) return false;
        return SquareMatrix::at(i, j);
    }

    void set(num_t i, num_t j, bool value) override {
        if (i > j) return;
        SquareMatrix::set(i, j, value);
    }
};
DAG CreateRandomIndegExpected(num_t N, double indeg, double sourceProb = 0.0) {
    DAG G(N);

    G.addDescriptionLine("HyperDAG from a random DAG with expected indegree " +
                         to_string(indeg) + ".");
    G.addDescriptionLine("Each node is chosen as a source with probability " +
                         to_string(sourceProb) + ".");

    for (num_t i = 1; i < N; ++i) {
        if ((double)rand() / (double)RAND_MAX < sourceProb) continue;

        double p = indeg / (double)i;

        for (num_t j = 0; j < i; ++j)
            if ((double)rand() / (double)RAND_MAX < p) G.addEdge(j, i);
    }

    return G;
}

DAG CreateRandomIndegFixed(num_t N, num_t indeg, double sourceProb = 0.0) {
    DAG G(N);

    G.addDescriptionLine("HyperDAG from a random DAG with fixed indegree " +
                         to_string(indeg) + ".");
    G.addDescriptionLine("Each node is chosen as a source with probability " +
                         to_string(sourceProb) + ".");

    for (num_t i = 1; i < N; ++i) {
        if ((double)rand() / (double)RAND_MAX < sourceProb) continue;

        if (i <= indeg)  // all previous nodes are chosen
            for (num_t j = 0; j < i; ++j) G.addEdge(j, i);

        else  // chose 'indeg' predecessors at random
        {
            vector<bool> chosen(i, false);
            for (num_t j = 0; j < indeg; ++j) {
                num_t rnd = randInt(i - j), idx = 0;
                for (; chosen[idx]; ++idx)
                    ;
                for (num_t k = 0; k < rnd; ++k) {
                    for (++idx; chosen[idx]; ++idx)
                        ;
                }

                G.addEdge(idx, i);
                chosen[idx] = true;
            }
        }
    }

    return G;
}

void CreateRandomER(DAG& G, num_t N, num_t NrOfEdges) {
    G.resize(N);
    G.addDescriptionLine(
        "HyperDAG from a random DAG with expected number of edges " +
        to_string(NrOfEdges) +
        ", with uniform edge probabilities on all forward edges");

    double p = 2.0 * (double)NrOfEdges / ((double)N * ((double)N - 1));

    for (num_t i = 0; i < N; ++i)
        for (num_t j = i + 1; j < N; ++j)
            if ((double)rand() / (double)RAND_MAX < p) G.addEdge(i, j);
}

/**
 * @brief Complete a DAG with an out-of-place SpMV operation.
 *
 * @param hyperdag DAG to extend, may be empty.
 * @param M        Matrix to multiply with vector v, which is
 *                 assumed to be of size M.n and dense.
 */
void CreateSpMV(DAG& G, const SquareMatrix& M, num_t vector_node_begin = 0) {
    if (DebugMode) M.print("SpMV");

    const num_t N = M.nrows();

    // Offsets
    const num_t source_dag_offset = G.size();
    const num_t v_offset_begin =
        vector_node_begin == 0 ? source_dag_offset : vector_node_begin;
    const num_t v_offset_end = v_offset_begin + N;
    const num_t M_offset_begin = v_offset_end;
    const num_t M_offset_end = M_offset_begin + N * N;
    const num_t mul_Mv_offset_begin = M_offset_end;
    const num_t mul_Mv_offset_end = mul_Mv_offset_begin + N * N;
    const num_t u_offset_begin = mul_Mv_offset_end;
    const num_t u_offset_end = u_offset_begin + N;

    // Create hyperDAG locally
    const num_t nNodes = u_offset_end;
    G.resize(G.size() + nNodes);
    G.addDescriptionLine("HyperDAG model of SpMV operation.");
    G.addDescriptionLine("Nodes of vector v: [" + to_string(v_offset_begin) +
                         ";" + to_string(v_offset_end - 1) + "]");
    G.addDescriptionLine("Nodes of matrix M: [" + to_string(M_offset_begin) +
                         ";" + to_string(M_offset_end - 1) + "] (row-wise)");
    G.addDescriptionLine("Nodes of the multiplication M * v: [" +
                         to_string(mul_Mv_offset_begin) + ";" +
                         to_string(mul_Mv_offset_end - 1) + "]");
    G.addDescriptionLine("Nodes of the final result u: [" +
                         to_string(u_offset_begin) + ";" +
                         to_string(u_offset_end - 1) + "]");
    std::stringstream strtream;
    M.print("M", strtream);
    G.addDescriptionLine(" ");
    G.addDescriptionLine(strtream.str());

    // find empty rows in the matrix
    vector<bool> rowNotEmpty(N, false);
    for (num_t i = 0; i < N; ++i)
        for (num_t j = 0; j < N; ++j)
            if (M.at(i, j)) rowNotEmpty[i] = true;

    // create SpMV DAG
    for (num_t i = 0; i < N; ++i) {
        if (!rowNotEmpty[i]) continue;

        for (num_t j = 0; j < N; ++j) {
            if (not M.at(i, j)) continue;

            // // Operation: u[j] += M[i][j] * v[i]
            // Nodes
            num_t node_M_i_j = M_offset_begin + i * N + j;
            num_t node_v_i = v_offset_begin + i;
            num_t node_mul_M_v = mul_Mv_offset_begin + i * N + j;
            num_t node_u_j = u_offset_begin + j;
            // Labels
            string M_str = "M[" + to_string(i) + "][" + to_string(j) + "]";
            string v_str = "v[" + to_string(i) + "]";
            string mul_M_v_str = "(" + M_str + " * " + v_str + ")";
            string u_str = "u[" + to_string(j) + "]";
            // Edges
            // M[i][j] * v[i] -> mul_M_v
            G.addEdge(node_M_i_j, node_mul_M_v, M_str + " -> " + mul_M_v_str);
            G.addEdge(node_v_i, node_mul_M_v, v_str + " -> " + mul_M_v_str);
            // mul_M_v -> u[j]
            G.addEdge(node_mul_M_v, node_u_j, mul_M_v_str + " -> " + u_str);
        }
    }

    // only keep components that are predecessors of the final nonzeros
    // (in particular, indeces corresponding to (i) empty columns in the
    // original vector or (ii) emtpy rows in the result vector)
    vector<num_t> sinkNodes;
    for (num_t i = 0; i < N; ++i)
        if (rowNotEmpty[i]) sinkNodes.push_back(u_offset_begin + i);
}

DAG CreateRandomSpMV(num_t N, double nonzero) {
    SquareMatrix M(N);
    M.randomize(nonzero);

    DAG G;
    CreateSpMV(G, M);
    return G;
}

void CreateLLtSolver(DAG& G, const LowerTriangularSquareMatrix& L) {
    std::cout << __PRETTY_FUNCTION__ << std::endl;

    if (DebugMode) {
        L.print("L");
    }

    const num_t n = L.nrows();
    const num_t nSquared = n * n;

    // Offsets
    const num_t offset = G.size();
    const num_t LOffset = offset;
    const num_t xOffset = LOffset + nSquared;
    const num_t y_MulOffset = xOffset + n;
    const num_t y_SubOffset = y_MulOffset + nSquared;
    const num_t yOffset = y_SubOffset + n;
    const num_t z_MulOffset = yOffset + n;
    const num_t z_SubOffset = z_MulOffset + nSquared;
    const num_t zOffset = z_SubOffset + n;

    const num_t nNodes = zOffset + n;
    G.resize(G.size() + nNodes);

    G.addDescriptionLine(
        "HyperDAG model of LLtSolver operation: inv(trans(L)) . (inv(L) . b)");
    G.addDescriptionLine("L.desc: " + L.getDescription());
    G.addDescriptionLine("Nodes of matrix L: [" + to_string(LOffset) + ";" +
                         to_string(LOffset + nSquared - 1) + "] (row-wise)");
    G.addDescriptionLine("Nodes of vector x: [" + to_string(xOffset) + ";" +
                         to_string(xOffset + n - 1) + "]");
    G.addDescriptionLine(
        "Nodes of the multiplication in the forward substitution "
        "phase to compute y: L[i][j] * x[j], range: [" +
        to_string(y_MulOffset) + ";" + to_string(y_MulOffset + nSquared - 1) +
        "]");
    G.addDescriptionLine(
        "Nodes of the subtraction in the forward substitution "
        "phase to compute y: y[i] - sum(L[i][j] * y[j]), range: [" +
        to_string(y_SubOffset) + ";" + to_string(y_SubOffset + n - 1) + "]");
    G.addDescriptionLine("Nodes of vector y: [" + to_string(yOffset) + ";" +
                         to_string(yOffset + n - 1) + "]");
    G.addDescriptionLine(
        "Nodes of the multiplication in the backward substitution "
        "phase to compute z: Lt[i][j] * z[j], range: [" +
        to_string(z_MulOffset) + ";" + to_string(z_MulOffset + nSquared - 1) +
        "]");
    G.addDescriptionLine(
        "Nodes of the subtraction in the forward substitution "
        "phase to compute z: z[i] - sum(Lt[i][j] * z[j]), range: [" +
        to_string(z_SubOffset) + ";" + to_string(z_SubOffset + n - 1) + "]");
    G.addDescriptionLine("Nodes of vector z: [" + to_string(zOffset) + ";" +
                         to_string(zOffset + n - 1) + "]");
    G.addDescriptionLine("Nodes of the final result z: [" + to_string(zOffset) +
                         ";" + to_string(zOffset + n - 1) + "]");

    std::stringstream strtream;
    L.print("L", strtream);
    G.addDescriptionLine(" ");
    G.addDescriptionLine(strtream.str());

    // find empty rows in the matrix L
    vector<bool> L_rowEmpty(n, true);
    for (num_t i = 0; i < n; ++i) {
        for (num_t j = 0; j < n; ++j)
            if (L.at(i, j)) {
                L_rowEmpty[i] = false;
                break;
            }
    }

    // Forward substitution DAG
    for (num_t i = 0; i < n; ++i) {
        if (L_rowEmpty[i]) continue;
        if (not L.at(i, i)) continue;
        if (DebugMode) cout << "-- row " << i << ":\n";

        const num_t y_i = yOffset + i;
        const string y_i_str = "y[" + to_string(i) + "]";

        const num_t x_i = xOffset + i;
        const string x_i_str = "x[" + to_string(i) + "]";

        const num_t L_i_i = LOffset + i * n + i;
        const string L_i_i_str =
            "L[" + to_string(i) + "][" + to_string(i) + "]";

        const num_t sub_x_Mul = y_SubOffset + i;
        const string sub_x_Mul_str =
            "(" + x_i_str + " - sum(y[j] * L[" + to_string(i) + "][j]))";

        for (num_t j = 0; j < i; ++j) {
            if (not L.at(i, j)) continue;

            const num_t L_i_j = LOffset + i * n + j;
            const string L_i_j_str =
                "L[" + to_string(i) + "][" + to_string(j) + "]";

            const num_t y_j = yOffset + j;
            const string y_j_str = "y[" + to_string(j) + "]";

            const num_t mul_L_x = y_MulOffset + i * n + j;
            const string mul_L_x_str = "(" + L_i_j_str + " * " + y_j_str + ")";

            G.addEdge(L_i_j, mul_L_x, L_i_j_str + " -> " + mul_L_x_str);
            G.addEdge(y_j, mul_L_x, y_j_str + " -> " + mul_L_x_str);

            G.addEdge(mul_L_x, sub_x_Mul, mul_L_x_str + " -> " + sub_x_Mul_str);
        }

        if (i > 0) {
            G.addEdge(sub_x_Mul, y_i, sub_x_Mul_str + " -> " + y_i_str);
            G.addEdge(x_i, sub_x_Mul, x_i_str + " -> " + sub_x_Mul_str);
        } else {
            G.addEdge(x_i, y_i, x_i_str + " -> " + y_i_str);
        }

        G.addEdge(L_i_i, y_i, L_i_i_str + " -> " + y_i_str);
    }
    if (DebugMode) cout << "-- Forward substitution DAG created." << endl;

    // find empty rows in the matrix L
    vector<bool> L_colEmpty(n, true);
    for (num_t j = 0; j < n; ++j) {
        for (num_t i = 0; i < n; ++i) {
            if (L.at(i, j)) {
                L_colEmpty[j] = false;
                break;
            }
        }
    }

    // Backward substitution DAG
    for (num_t _i = n; _i > 0; --_i) {
        num_t i = _i - 1;
        if (L_colEmpty[i]) continue;
        if (not L.at(i, i)) continue;
        if (DebugMode) cout << "-- column " << i << ":\n";

        const num_t z_i = zOffset + i;
        const string z_i_str = "z[" + to_string(i) + "]";

        const num_t y_i = yOffset + i;
        const string y_i_str = "y[" + to_string(i) + "]";

        const num_t Lt_i_i = LOffset + i * n + i;
        const string Lt_i_i_str =
            "Lt[" + to_string(i) + "][" + to_string(i) + "]";

        const num_t sub_x_Mul = z_SubOffset + i;
        const string sub_x_Mul_str =
            "(" + y_i_str + " - sum(z[j] * Lt[" + to_string(i) + "][j]))";

        for (num_t _j = n; _j > i + 1; --_j) {
            num_t j = _j - 1;
            if (not L.at(j, i)) continue;

            const num_t Lt_i_j = LOffset + j * n + i;
            const string Lt_i_j_str =
                "Lt[" + to_string(i) + "][" + to_string(j) + "]";

            const num_t z_j = zOffset + j;
            const string z_j_str = "z[" + to_string(j) + "]";

            const num_t mul_L_x = z_MulOffset + i * n + j;
            const string mul_L_x_str = "(" + Lt_i_j_str + " * " + z_j_str + ")";

            G.addEdge(Lt_i_j, mul_L_x, Lt_i_j_str + " -> " + mul_L_x_str);
            G.addEdge(z_j, mul_L_x, z_j_str + " -> " + mul_L_x_str);

            G.addEdge(mul_L_x, sub_x_Mul, mul_L_x_str + " -> " + sub_x_Mul_str);
        }

        if (i < n - 1) {
            G.addEdge(sub_x_Mul, z_i, sub_x_Mul_str + " -> " + z_i_str);
            G.addEdge(y_i, sub_x_Mul, y_i_str + " -> " + sub_x_Mul_str);
        } else {
            G.addEdge(y_i, z_i, y_i_str + " -> " + z_i_str);
        }

        G.addEdge(Lt_i_i, z_i, Lt_i_i_str + " -> " + z_i_str);
    }
    if (DebugMode) cout << "-- Backward substitution DAG created." << endl;
}

void CreateALLtSolver(DAG& G, const SquareMatrix& A,
                      const LowerTriangularSquareMatrix& L) {
    if (DebugMode) {
        A.print("ALLtSolver: A");
        L.print("ALLtSolver: L");
    }
    CreateLLtSolver(G, L);
    CreateSpMV(G, A, G.size() - L.nrows());
}

void CreateLUSolver(DAG& G, const LowerTriangularSquareMatrix& L,
                    const UpperTriangularSquareMatrix& U) {
    if (DebugMode) {
        L.print("LUSolver: L");
        U.print("LUSolver: U");
    }

    if (L.nrows() != U.nrows()) {
        cerr << "Error: L and U matrices must have the same size." << endl;
        abort();
    }

    const num_t n = L.nrows();
    const num_t nSquared = n * n;

    // Offsets
    const num_t offset = G.size();
    const num_t LOffset = offset;
    const num_t UOffset = LOffset + nSquared;
    const num_t xOffset = UOffset + nSquared;
    const num_t y_MulOffset = xOffset + n;
    const num_t y_SubOffset = y_MulOffset + nSquared;
    const num_t yOffset = y_SubOffset + n;
    const num_t z_MulOffset = yOffset + n;
    const num_t z_SubOffset = z_MulOffset + nSquared;
    const num_t zOffset = z_SubOffset + n;

    const num_t nNodes = zOffset + n;
    G.resize(G.size() + nNodes);

    G.addDescriptionLine(
        "HyperDAG model of LUSolver operation: inv(U) . (inv(L) . b)");
    G.addDescriptionLine("L.desc: " + L.getDescription());
    G.addDescriptionLine("U.desc: " + U.getDescription());
    G.addDescriptionLine("Nodes of matrix L: [" + to_string(LOffset) + ";" +
                         to_string(LOffset + nSquared - 1) + "] (row-wise)");
    G.addDescriptionLine("Nodes of matrix U: [" + to_string(UOffset) + ";" +
                         to_string(UOffset + nSquared - 1) + "] (row-wise)");
    G.addDescriptionLine("Nodes of vector x: [" + to_string(xOffset) + ";" +
                         to_string(xOffset + n - 1) + "]");
    G.addDescriptionLine(
        "Nodes of the multiplication in the forward substitution "
        "phase to compute y: L[i][j] * x[j], range: [" +
        to_string(y_MulOffset) + ";" + to_string(y_MulOffset + nSquared - 1) +
        "]");
    G.addDescriptionLine(
        "Nodes of the subtraction in the forward substitution "
        "phase to compute y: y[i] - sum(L[i][j] * y[j]), range: [" +
        to_string(y_SubOffset) + ";" + to_string(y_SubOffset + n - 1) + "]");
    G.addDescriptionLine("Nodes of vector y: [" + to_string(yOffset) + ";" +
                         to_string(yOffset + n - 1) + "]");
    G.addDescriptionLine(
        "Nodes of the multiplication in the backward substitution "
        "phase to compute z: U[i][j] * z[j], range: [" +
        to_string(z_MulOffset) + ";" + to_string(z_MulOffset + nSquared - 1) +
        "]");
    G.addDescriptionLine(
        "Nodes of the subtraction in the forward substitution "
        "phase to compute z: z[i] - sum(U[i][j] * z[j]), range: [" +
        to_string(z_SubOffset) + ";" + to_string(z_SubOffset + n - 1) + "]");
    G.addDescriptionLine("Nodes of vector z: [" + to_string(zOffset) + ";" +
                         to_string(zOffset + n - 1) + "]");
    G.addDescriptionLine("Nodes of the final result z: [" + to_string(zOffset) +
                         ";" + to_string(zOffset + n - 1) + "]");

    // find empty rows in the matrix L and U
    vector<bool> L_rowEmpty(n, true), U_rowEmpty(n, true);
    for (num_t i = 0; i < n; ++i) {
        for (num_t j = 0; j < n; ++j) {
            if (L.at(i, j)) {
                L_rowEmpty[i] = false;
                break;
            }
        }
        for (num_t j = 0; j < n; ++j) {
            if (U.at(i, j)) {
                U_rowEmpty[i] = false;
                break;
            }
        }
    }

    // Forward substitution DAG
    for (num_t i = 0; i < n; ++i) {
        if (L_rowEmpty[i]) continue;
        if (not L.at(i, i)) continue;
        if (DebugMode) cout << "-- row " << i << ":\n";

        const num_t y_i = yOffset + i;
        const string y_i_str = "y[" + to_string(i) + "]";

        const num_t x_i = xOffset + i;
        const string x_i_str = "x[" + to_string(i) + "]";

        const num_t L_i_i = LOffset + i * n + i;
        const string L_i_i_str =
            "L[" + to_string(i) + "][" + to_string(i) + "]";

        const num_t sub_x_Mul = y_SubOffset + i;
        const string sub_x_Mul_str =
            "(" + x_i_str + " - sum(y[j] * L[" + to_string(i) + "][j]))";

        for (num_t j = 0; j < i; ++j) {
            if (not L.at(i, j)) continue;

            const num_t L_i_j = LOffset + i * n + j;
            const string L_i_j_str =
                "L[" + to_string(i) + "][" + to_string(j) + "]";

            const num_t y_j = yOffset + j;
            const string y_j_str = "y[" + to_string(j) + "]";

            const num_t mul_L_x = y_MulOffset + i * n + j;
            const string mul_L_x_str = "(" + L_i_j_str + " * " + y_j_str + ")";

            G.addEdge(L_i_j, mul_L_x, L_i_j_str + " -> " + mul_L_x_str);
            G.addEdge(y_j, mul_L_x, y_j_str + " -> " + mul_L_x_str);

            G.addEdge(mul_L_x, sub_x_Mul, mul_L_x_str + " -> " + sub_x_Mul_str);
        }

        if (i > 0) {
            G.addEdge(sub_x_Mul, y_i, sub_x_Mul_str + " -> " + y_i_str);
            G.addEdge(x_i, sub_x_Mul, x_i_str + " -> " + sub_x_Mul_str);
        } else {
            G.addEdge(x_i, y_i, x_i_str + " -> " + y_i_str);
        }

        G.addEdge(L_i_i, y_i, L_i_i_str + " -> " + y_i_str);
    }
    if (DebugMode) cout << "-- Forward substitution DAG created." << endl;

    // Backward substitution DAG
    for (num_t _i = n; _i > 0; --_i) {
        num_t i = _i - 1;
        if (U_rowEmpty[i]) continue;
        if (DebugMode) cout << "-- row " << i << ":\n";

        const num_t z_i = zOffset + i;
        const string z_i_str = "z[" + to_string(i) + "]";

        const num_t y_i = yOffset + i;
        const string y_i_str = "y[" + to_string(i) + "]";

        const num_t U_i_i = UOffset + i * n + i;
        const string U_i_i_str =
            "U[" + to_string(i) + "][" + to_string(i) + "]";

        const num_t sub_x_Mul = z_SubOffset + i;
        const string sub_x_Mul_str =
            "(" + y_i_str + " - sum(z[j] * U[" + to_string(i) + "][j]))";

        for (num_t _j = n; _j > i + 1; --_j) {
            num_t j = _j - 1;
            if (not U.at(i, j)) continue;

            const num_t U_i_j = UOffset + i * n + j;
            const string U_i_j_str =
                "U[" + to_string(i) + "][" + to_string(j) + "]";

            const num_t z_j = zOffset + j;
            const string z_j_str = "z[" + to_string(j) + "]";

            const num_t mul_U_x = z_MulOffset + i * n + j;
            const string mul_U_x_str = "(" + U_i_j_str + " * " + z_j_str + ")";

            G.addEdge(U_i_j, mul_U_x, U_i_j_str + " -> " + mul_U_x_str);
            G.addEdge(z_j, mul_U_x, z_j_str + " -> " + mul_U_x_str);

            G.addEdge(mul_U_x, sub_x_Mul, mul_U_x_str + " -> " + sub_x_Mul_str);
        }

        if (i == n - 1) {
            G.addEdge(y_i, z_i, y_i_str + " -> " + z_i_str);
        } else {
            G.addEdge(y_i, sub_x_Mul, y_i_str + " -> " + sub_x_Mul_str);
            G.addEdge(sub_x_Mul, z_i, sub_x_Mul_str + " -> " + z_i_str);
        }

        G.addEdge(U_i_i, z_i, U_i_i_str + " -> " + z_i_str);
    }
    if (DebugMode) cout << "-- Backward substitution DAG created." << endl;
}

DAG CreateRandomLUSolver(num_t N, double nonzero) {
    LowerTriangularSquareMatrix L(N);
    UpperTriangularSquareMatrix U(N);
    L.randomize(nonzero);
    U.randomize(nonzero);
    DAG G;
    CreateLUSolver(G, L, U);
    return G;
}

void CreateALUSolver(DAG& G, const SquareMatrix& A,
                     const LowerTriangularSquareMatrix& L,
                     const UpperTriangularSquareMatrix& U) {
    if (DebugMode) {
        L.print("ALUSolver: L");
        U.print("ALUSolver: U");
        A.print("ALUSolver: A");
    }
    CreateLUSolver(G, L, U);
    CreateSpMV(G, A, G.size() - L.nrows());
}

DAG CreateRandomALUSolver(num_t N, double nonzero) {
    SquareMatrix A(N);
    LowerTriangularSquareMatrix L(N);
    UpperTriangularSquareMatrix U(N);
    A.randomize(nonzero);
    L.randomize(nonzero);
    U.randomize(nonzero);
    DAG G;
    CreateALUSolver(G, A, L, U);
    return G;
}

DAG CreateSpMVExp(const SquareMatrix& M, num_t K) {
    num_t N = M.nrows();

    DAG G((K + 1) * (M.nnz() + N));
    G.addDescriptionLine("HyperDAG model of naive implementation of A^" +
                         to_string(K) +
                         " *v with sparse matrix A and dense vector v.");
    G.addDescriptionLine(M.getDescription());

    // Index of the nodes representing each nonzero of M
    vector<bool> rowValid(N, true), colValid(N, true);
    vector<vector<num_t>> originalCellIdx(N, vector<num_t>(N));
    num_t Idx = 0;
    for (num_t i = 0; i < N; ++i) {
        for (num_t j = 0; j < N; ++j) {
            if (M.at(i, j)) {
                originalCellIdx[i][j] = Idx;
                ++Idx;
            }
        }
    }

    // Vbase is the starting point of the current vector v, base is the start
    // index for the next iteration
    num_t Vbase = M.nnz(), base = Vbase + N;

    // ITERATIONS
    for (num_t k = 0; k < K; ++k) {
        colValid = rowValid;

        // count the number of used cells in current iteration
        num_t usedCells = 0;
        for (num_t i = 0; i < N; ++i)
            for (num_t j = 0; j < N; ++j)
                if (M.at(i, j) && colValid[j]) ++usedCells;

        if (usedCells == 0) {
            cout << "Error: the whole matrix has vanished (we get a zero "
                    "vector after "
                 << k + 1 << " iterations)!\n";
            cout << "The output hyperDAG corresponds to the result after " << k
                 << " iterations.\n";
            break;
        }

        // iterate over cells, add DAG edges
        Idx = 0;
        rowValid.assign(N, false);
        for (num_t i = 0; i < N; ++i)
            for (num_t j = 0; j < N; ++j)
                if (M.at(i, j) && colValid[j]) {
                    G.addEdge(Vbase + j, base + Idx);
                    G.addEdge(originalCellIdx[i][j], base + Idx);
                    G.addEdge(base + Idx, base + usedCells + i);
                    ++Idx;
                    rowValid[i] = true;
                }

        Vbase = base + usedCells;
        base = Vbase + N;
    }

    // only keep components that are predecessors of the final nonzeros
    // (this can remove many intermediate values that were never used: they were
    // computed, but then they were only multiplied by zeros)
    vector<num_t> finalVector;
    for (num_t i = 0; i < N; ++i)
        if (rowValid[i]) finalVector.push_back(Vbase + i);

    return G.keepGivenNodes(G.isReachable(finalVector, true));
}

DAG CreateRandomSpMVExp(num_t N, double nonzero, num_t K) {
    SquareMatrix M(N);
    M.randomize(nonzero);

    DAG G = CreateSpMVExp(M, K);

    return G;
}

DAG CreateSparseCG(const SquareMatrix& M, num_t K) {
    num_t N = M.nrows();

    DAG G(3 * N - 1 + 2 * M.nnz() + K * (M.nnz() + 9 * N + 4));
    G.addDescriptionLine("HyperDAG model of naive implementation of " +
                         to_string(K) +
                         " iterations of the conjugate gradient method.");
    G.addDescriptionLine(M.getDescription());

    // Index of the nodes representing each nonzero of M
    // (we will denote the matrix M by "A" in the comment pseudocodes, since it
    // is more standard)
    vector<vector<num_t>> originalCellIdx(N, vector<num_t>(N));
    num_t Idx = 0;
    for (num_t i = 0; i < N; ++i)
        for (num_t j = 0; j < N; ++j)
            if (M.at(i, j)) {
                originalCellIdx[i][j] = Idx;
                ++Idx;
            }

    // INITIALIZATION PHASE

    // compute A * x_0
    Idx = 0;
    for (num_t i = 0; i < N; ++i)
        for (num_t j = 0; j < N; ++j) {
            if (M.at(i, j)) {
                G.addEdge(M.nnz() + j, M.nnz() + N + Idx);
                G.addEdge(originalCellIdx[i][j], M.nnz() + N + Idx);
                G.addEdge(M.nnz() + N + Idx, 2 * M.nnz() + N + i);
                Idx += 1;
            }
        }

    // Rbase, Pbase and Xbase points to the beginning of the current r, p and x
    // vectors
    num_t Rbase = 2 * M.nnz() + 3 * N, Pbase = Rbase + N, Xbase = M.nnz();

    // r_0 := b - A * x_0 ;  p_0 := r_0
    for (num_t i = 0; i < N; ++i) {
        G.addEdge(2 * M.nnz() + N + i, Rbase + i);
        G.addEdge(2 * M.nnz() + 2 * N + i, Rbase + i);
        G.addEdge(Rbase + i, Pbase + i);
    }

    // compute r_0^T * r_0 and save its index in Rproduct
    num_t Rproduct = Pbase + 2 * N;
    for (num_t i = 0; i < N; ++i) {
        G.addEdge(Rbase + i, Pbase + N + i);
        G.addEdge(Pbase + N + i, Rproduct);
    }

    // ITERATIONS

    // base denotes the current index at the beginning of each iteration
    num_t base = Rproduct + 1;

    for (num_t k = 0; k < K; ++k) {
        // compute A * p_k
        Idx = 0;
        for (num_t i = 0; i < N; ++i)
            for (num_t j = 0; j < N; ++j) {
                if (M.at(i, j)) {
                    G.addEdge(Pbase + j, base + Idx);
                    G.addEdge(originalCellIdx[i][j], base + Idx);
                    G.addEdge(base + Idx, base + M.nnz() + i);
                    Idx += 1;
                }
            }

        // compute p_k^T * A * p_k
        for (num_t i = 0; i < N; ++i) {
            G.addEdge(Pbase + i, base + M.nnz() + N + i);
            G.addEdge(base + M.nnz() + i, base + M.nnz() + N + i);
            G.addEdge(base + M.nnz() + N + i, base + M.nnz() + 2 * N);
        }

        // alpha_k := r_k^T * r_k / p_k^T * A * p_k
        G.addEdge(base + M.nnz() + 2 * N, base + M.nnz() + 2 * N + 1);
        G.addEdge(Rproduct, base + M.nnz() + 2 * N + 1);

        // index of alpha_k
        num_t alpha = base + M.nnz() + 2 * N + 1;

        // x_(k+1) := x_k + alpha_k * p_k
        for (num_t i = 0; i < N; ++i) {
            G.addEdge(Pbase + i, alpha + 1 + i);
            G.addEdge(alpha, alpha + 1 + i);
            G.addEdge(alpha + 1 + i, alpha + 1 + N + i);
            G.addEdge(Xbase + i, alpha + 1 + N + i);
        }
        Xbase = alpha + 1 + N;

        // r_(k+1) := r_k - alpha_k * A * p_k
        for (num_t i = 0; i < N; ++i) {
            G.addEdge(base + M.nnz() + i, Xbase + N + i);
            G.addEdge(alpha, Xbase + N + i);
            G.addEdge(Xbase + N + i, Xbase + 2 * N + i);
            G.addEdge(Rbase + i, Xbase + 2 * N + i);
        }
        Rbase = Xbase + 2 * N;

        if (k == K - 1) break;

        // compute r_(k+1)^T * r_(k+1)
        for (num_t i = 0; i < N; ++i) {
            G.addEdge(Rbase + i, Rbase + N + i);
            G.addEdge(Rbase + N + i, Rbase + 2 * N);
        }

        // beta_k := r_(k+1)^T * r_(k+1) / r_k^T * r_k
        G.addEdge(Rbase + 2 * N, Rbase + 2 * N + 1);
        G.addEdge(Rproduct, Rbase + 2 * N + 1);

        // updated index of Rproduct and beta_k
        Rproduct = Rbase + 2 * N;
        num_t beta = Rbase + 2 * N + 1;

        // p_(k+1) := r_(k+1) - beta_k * p_k
        for (num_t i = 0; i < N; ++i) {
            G.addEdge(Pbase + i, beta + 1 + i);
            G.addEdge(beta, beta + 1 + i);
            G.addEdge(beta + 1 + i, beta + 1 + N + i);
            G.addEdge(Rbase + i, beta + 1 + N + i);
        }

        Pbase = beta + 1 + N;
        base = Pbase + N;
    }

    return G;
}

DAG CreateRandomSparseCG(num_t N, double nonzero, num_t K) {
    SquareMatrix M(N);
    M.randomize(nonzero);

    DAG G = CreateSparseCG(M, K);

    return G;
}

DAG CreatekNN(const SquareMatrix& M, num_t K, num_t source) {
    num_t N = M.nrows();

    DAG G((K + 1) * (M.nnz() + N));
    G.addDescriptionLine(
        "HyperDAG model of naive implementation of " + to_string(K) +
        " iterations of kNN, starting from node number " + to_string(source) +
        " (i.e. sparse vector with 1 entry).");
    G.addDescriptionLine(M.getDescription());

    // Index of the nodes representing each nonzero of M
    vector<vector<num_t>> originalCellIdx(N, vector<num_t>(N));
    num_t Idx = 0;
    for (num_t i = 0; i < N; ++i)
        for (num_t j = 0; j < N; ++j)
            if (M.at(i, j)) {
                originalCellIdx[i][j] = Idx;
                ++Idx;
            }

    // Initialize sparse vector
    vector<bool> reached(N, false);
    reached[source] = true;

    // Vbase is the current index of the beginning of the vector
    num_t Vbase = Idx;
    // base is the index where the next iteration starts
    num_t base = Idx + 1;

    // current new index for each original row of the sparse vector
    vector<num_t> newRowIdx(N);
    newRowIdx[source] = 0;

    // Iterations
    for (num_t k = 0; k < K; ++k) {
        // count the number of internal nodes (matrix-vector multiplication) in
        // this iteration
        num_t NrOfInternalNodes = 0;
        for (num_t i = 0; i < N; ++i)
            for (num_t j = 0; j < N; ++j)
                if (reached[j] && M.at(i, j)) ++NrOfInternalNodes;

        // add DAG edges
        num_t rowIdx = 0, cellIdx = 0;
        vector<bool> rowNotEmpty(N, false);
        for (num_t i = 0; i < N; ++i) {
            for (num_t j = 0; j < N; ++j)
                if (reached[j] && M.at(i, j)) {
                    G.addEdge(Vbase + newRowIdx[j], base + cellIdx);
                    G.addEdge(originalCellIdx[i][j], base + cellIdx);
                    G.addEdge(base + cellIdx,
                              base + NrOfInternalNodes + rowIdx);
                    cellIdx += 1;
                    rowNotEmpty[i] = true;
                }

            if (rowNotEmpty[i]) ++rowIdx;
        }

        // setup for next iteration
        rowIdx = 0;
        for (num_t i = 0; i < N; ++i)
            if (rowNotEmpty[i]) {
                reached[i] = true;
                newRowIdx[i] = rowIdx;
                ++rowIdx;
            }

        num_t oldbase = base;
        base = oldbase + NrOfInternalNodes + rowIdx;
        Vbase = oldbase + NrOfInternalNodes;
    }

    // only keep components that are connected to the final nonzeros
    // (in particular, this removes nonzeros of the matrix that were never used)
    vector<num_t> finalVector;
    for (num_t i = Vbase; i < base; ++i) finalVector.push_back(i);

    return G.keepGivenNodes(G.isReachable(finalVector));
}

DAG CreateRandomkNN(num_t N, double nonzero, num_t K) {
    SquareMatrix M(N);
    M.randomize(nonzero);
    M.setMainDiagonal();

    num_t source = randInt(N);

    DAG G = CreatekNN(M, K, source);

    return G;
}

class Parser {
   public:
    string mode = "";

    bool has_N = false;
    size_t N = 0;

    bool has_nonzeroProb = false;
    double nonzeroProb = 0.0;

    bool has_K = false;
    size_t K = 0;

    bool has_edges = false;
    size_t edges = 0;

    bool has_indegree = false;
    double indegree = .0;

    bool has_sourceProb = false;
    double sourceProb = 0.0;

    bool has_sourceNode = true;
    size_t sourceNode = 0;

    bool has_inputs = false;
    vector<string> inputs;

    bool has_output = false;
    string output = "output.mtx";

    Parser(int argc, char** argv) {
        if (argc <= 1) {
            showUsage();
            exit(1);
        }
        mode = str_tolower(argv[1]);
        for (int i = 2; i < argc; ++i) {
            const string arg = argv[i];
            // Check for: -debug
            if (arg == "-debug") {
                DebugMode = true;
                continue;
            }
            // Check for: -N <value>
            if (arg == "-N") {
                if (i + 1 < argc) {
                    has_N = true;
                    N = stoul(argv[++i]);
                }
                continue;
            }
            // Check for: -nonzeroProb <value>
            if (arg == "-nonzeroProb") {
                if (i + 1 < argc) {
                    has_nonzeroProb = true;
                    nonzeroProb = stod(argv[++i]);
                }
                continue;
            }
            // Check for: -K <value>
            if (arg == "-K") {
                if (i + 1 < argc) {
                    has_K = true;
                    K = stoul(argv[++i]);
                }
                continue;
            }
            // Check for: -edges <value>
            if (arg == "-edges") {
                if (i + 1 < argc) {
                    has_edges = true;
                    edges = stoul(argv[++i]);
                }
                continue;
            }
            // Check for: -indegree <value>
            if (arg == "-indegree") {
                if (i + 1 < argc) {
                    has_indegree = true;
                    indegree = stod(argv[++i]);
                }
                continue;
            }
            // Check for: -sourceProb <value>
            if (arg == "-sourceProb") {
                if (i + 1 < argc) {
                    has_sourceProb = true;
                    sourceProb = stod(argv[++i]);
                }
                continue;
            }
            // Check for: -sourceNode <value>
            if (arg == "-sourceNode") {
                if (i + 1 < argc) {
                    has_sourceNode = true;
                    sourceNode = stoul(argv[++i]);
                }
                continue;
            }
            // Check for: -output <filename>
            if (arg == "-output") {
                if (i + 1 < argc) {
                    has_output = true;
                    output = argv[++i];
                }
                continue;
            }
            // Check for: -inputs <value1> <value2> ...
            if (arg == "-inputs") {
                while (i + 1 < argc && argv[i + 1][0] != '-') {
                    has_inputs = true;
                    inputs.push_back(argv[++i]);
                }
                continue;
            }
        }

        // Check that a mode was provided
        if (mode.empty() || std::find(getModes().cbegin(), getModes().cend(),
                                      mode) == getModes().cend()) {
            showUsage("No or invalid mode provided: " + mode + "\n");
            exit(1);
        }
        // Check that N (if provided) is higher than 1
        if (has_N && N <= 1) {
            showUsage("<N> must be >= 2.\n");
            exit(1);
        }
        // Check that N is provided for modes that require it, if not provided
        // set to default
        if (not has_N &&
            (mode == "er" || mode == "fixedin" || mode == "expectedin")) {
            showUsage("No <N> provided, required by this mode: -N <value>\n");
            exit(1);
        } else if (not has_N) {
            has_N = true;
            N = 10;
        }
        // Check that nonzeroProb (if provided) is between 0 and 1, if not
        // provided set to default
        if (has_nonzeroProb && (nonzeroProb < 0.0 || nonzeroProb > 1.0)) {
            showUsage("<nonzeroProb> must be in the range [0.0;1.0]\n");
            exit(1);
        } else if (not has_nonzeroProb) {
            has_nonzeroProb = true;
            nonzeroProb = .5;
        }
        // Check that K (if provided) is higher than 0, or if not provided set
        // to default
        if (has_K && K <= 0) {
            showUsage("<K> must be >= 1.\n");
            exit(1);
        } else if (not has_K) {
            has_K = true;
            K = 3;
        }
        // Check that edges (if provided) is higher than 0, if not provided set
        // to default
        if (has_edges && edges <= 0) {
            showUsage("<edges> must be >= 1.\n");
            exit(1);
        } else if (not has_edges) {
            has_edges = true;
            edges = N * (N - 1) / 5;
        }
        // Check that sourceProb (if provided) is between 0 and 1, if not
        // provided set to default
        if (has_sourceProb && (sourceProb < 0.0 || sourceProb > 1.0)) {
            showUsage("<sourceProb> must be in the range [0.0;1.0]\n");
            exit(1);
        } else if (not has_sourceProb) {
            has_sourceProb = true;
            sourceProb = .0;
        }
        // Check that has_indegree (if provided) is higher than 0, if not
        // provided set to default
        if (has_indegree && has_N && (indegree < 1 || indegree > N - 1)) {
            showUsage("<indegree> must be in the range [1;N-1]\n");
            exit(1);
        } else if (not has_indegree) {
            has_indegree = true;
            indegree = 4;
        }
        // Check that output (if provided) is not empty
        if (has_output && output.empty()) {
            showUsage("<output> must be a valid filename.\n");
            exit(1);
        }
        // Check that inputs (if provided) is not empty
        if (has_inputs &&
            (inputs.empty() ||
             std::any_of(inputs.cbegin(), inputs.cend(),
                         [](const string& s) { return s.empty(); }))) {
            showUsage("<inputs> must contain valid filenames.\n");
            exit(1);
        }
    }

    void showUsage(string message = "") const {
        if (!message.empty()) {
            cerr << "Error: " << message << endl;
        }
        cerr << "Usage: " << endl;
        cerr << "  "
             << "help <mode>" << endl;
        for (const auto& entry : modes_usage) {
            cerr << "  " << entry.first << " " << entry.second << endl;
        }
    }
};

namespace command {

static void help(const Parser& arguments, DAG& G) {
    if (arguments.mode.empty())
        cerr << "Unknown mode\n";
    else
        cerr << "Usage of mode " << arguments.mode << ": "
             << modes_usage.at(arguments.mode) << " <mode> <mode_args>\n";
    exit(0);
}

static void ER(const Parser& arguments, DAG& G) {
    assert(arguments.has_N);
    assert(arguments.has_edges);

    CreateRandomER(G, arguments.N, arguments.edges);
}

static void SpMV(const Parser& arguments, DAG& G) {
    std::unique_ptr<SquareMatrix> A = nullptr;
    if (arguments.has_inputs) {
        A = make_unique<SquareMatrix>(
            IMatrix::readFromFile<SquareMatrix>(arguments.inputs[0]));
    } else {
        assert(arguments.has_N);
        assert(arguments.has_nonzeroProb);
        A = make_unique<SquareMatrix>(arguments.N);
        A->randomize(arguments.nonzeroProb);
    }
    CreateSpMV(G, *A);
}

static void LLtSolver(const Parser& arguments, DAG& G) {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    std::unique_ptr<LowerTriangularSquareMatrix> L;
    if (arguments.has_inputs) {
        L = make_unique<LowerTriangularSquareMatrix>(
            IMatrix::readFromFile<LowerTriangularSquareMatrix>(
                arguments.inputs[0]));
    } else {
        assert(arguments.has_N);
        assert(arguments.has_nonzeroProb);
        L = make_unique<LowerTriangularSquareMatrix>(arguments.N);
        L->randomize(arguments.nonzeroProb);
    }
    CreateLLtSolver(G, *L);
}

static void LLtSolverExp(const Parser& arguments, DAG& G) {
    throw runtime_error("Not implemented yet");
    assert(arguments.has_K);
}

static void ALLtSolver(const Parser& arguments, DAG& G) {
    unique_ptr<SquareMatrix> A;
    unique_ptr<LowerTriangularSquareMatrix> L;
    if (arguments.has_inputs) {
        A = make_unique<SquareMatrix>(
            IMatrix::readFromFile<SquareMatrix>(arguments.inputs[0]));
        L = make_unique<LowerTriangularSquareMatrix>(
            IMatrix::readFromFile<LowerTriangularSquareMatrix>(
                arguments.inputs[1]));
    } else {
        A = make_unique<SquareMatrix>(arguments.N);
        A->randomize(arguments.nonzeroProb);
        L = make_unique<LowerTriangularSquareMatrix>(arguments.N);
        L->randomize(arguments.nonzeroProb);
    }
    CreateALLtSolver(G, *A, *L);
}

static void ALLtSolverExp(const Parser& arguments, DAG& G) {
    throw runtime_error("Not implemented yet");
    assert(arguments.has_K);
}

static void LUSolver(const Parser& arguments, DAG& G) {
    unique_ptr<LowerTriangularSquareMatrix> L;
    unique_ptr<UpperTriangularSquareMatrix> U;
    if (arguments.has_inputs) {
        L = make_unique<LowerTriangularSquareMatrix>(
            IMatrix::readFromFile<LowerTriangularSquareMatrix>(
                arguments.inputs[0]));
        U = make_unique<UpperTriangularSquareMatrix>(
            IMatrix::readFromFile<UpperTriangularSquareMatrix>(
                arguments.inputs[1]));
    } else {
        L = make_unique<LowerTriangularSquareMatrix>(arguments.N);
        L->randomize(arguments.nonzeroProb);
        U = make_unique<UpperTriangularSquareMatrix>(arguments.N);
        U->randomize(arguments.nonzeroProb);
    }
    CreateLUSolver(G, *L, *U);
}

static void LUSolverExp(const Parser& arguments, DAG& G) {
    throw runtime_error("Not implemented yet");
    assert(arguments.has_K);
}

static void ALUSolver(const Parser& arguments, DAG& G) {
    unique_ptr<SquareMatrix> A;
    unique_ptr<LowerTriangularSquareMatrix> L;
    unique_ptr<UpperTriangularSquareMatrix> U;
    if (arguments.has_inputs) {
        A = make_unique<SquareMatrix>(
            IMatrix::readFromFile<SquareMatrix>(arguments.inputs[0]));
        L = make_unique<LowerTriangularSquareMatrix>(
            IMatrix::readFromFile<LowerTriangularSquareMatrix>(
                arguments.inputs[1]));
        U = make_unique<UpperTriangularSquareMatrix>(
            IMatrix::readFromFile<UpperTriangularSquareMatrix>(
                arguments.inputs[2]));
    } else {
        A = make_unique<SquareMatrix>(arguments.N);
        A->randomize(arguments.nonzeroProb);
        L = make_unique<LowerTriangularSquareMatrix>(arguments.N);
        L->randomize(arguments.nonzeroProb);
        U = make_unique<UpperTriangularSquareMatrix>(arguments.N);
        U->randomize(arguments.nonzeroProb);
    }
    CreateALUSolver(G, *A, *L, *U);
}

static void ALUSolverExp(const Parser& arguments, DAG& G) {
    throw runtime_error("Not implemented yet");
    assert(arguments.has_K);
}

static void fixedIn(const Parser& arguments, DAG& G) {
    assert(arguments.has_indegree);
    assert(arguments.indegree >= 1.0);
}

static void expectedIn(const Parser& arguments, DAG& G) {
    assert(arguments.has_indegree);
    assert(arguments.indegree >= .0);
}

static void kNN(const Parser& arguments, DAG& G) {
    assert(arguments.has_sourceNode);
    assert(arguments.sourceNode >= 0);
    assert(arguments.sourceNode < arguments.N);
}

static void CG(const Parser& arguments, DAG& G) {
    assert(arguments.has_sourceNode);
    assert(arguments.sourceNode >= 0);
    assert(arguments.sourceNode < arguments.N);
}

}  // namespace command

int main(int argc, char** argv) {
    srand(0);
    const Parser arguments(argc, argv);

    DAG G;
    if (arguments.mode == "help") {
        command::help(arguments, G);
        return 0;
    } else if (arguments.mode == "er") {
        command::ER(arguments, G);
    } else if (arguments.mode == "spmv") {
        command::SpMV(arguments, G);
    } else if (arguments.mode == "lltsolver") {
        command::LLtSolver(arguments, G);
    } else if (arguments.mode == "lltsolverexp") {
        command::LLtSolverExp(arguments, G);
    } else if (arguments.mode == "alltsolver") {
        command::ALLtSolver(arguments, G);
    } else if (arguments.mode == "alltsolverexp") {
        command::ALLtSolverExp(arguments, G);
    } else if (arguments.mode == "lusolver") {
        command::LUSolver(arguments, G);
    } else if (arguments.mode == "lusolverexp") {
        command::LUSolverExp(arguments, G);
    } else if (arguments.mode == "alusolver") {
        command::ALUSolver(arguments, G);
    } else if (arguments.mode == "alusolverexp") {
        command::ALUSolverExp(arguments, G);
    } else if (arguments.mode == "fixedin") {
        command::fixedIn(arguments, G);
    } else if (arguments.mode == "expectedin") {
        command::expectedIn(arguments, G);
    } else if (arguments.mode == "knn") {
        command::kNN(arguments, G);
    } else if (arguments.mode == "cg") {
        command::CG(arguments, G);
    } else {
        cerr << "Unknown mode: " << arguments.mode << endl;
        return 1;
    }

    G.printDAG(arguments.output + ".dag");
    G.printHyperDAG(arguments.output);
    return 0;
}
