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
#include <cmath>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

bool DebugMode = false;

// AUXILIARY FUNCTIONS

// unbiased random int generator
int randInt(int lim) {
    int rnd = rand();
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
    int n;
    vector<vector<int> > In, Out;
    vector<string> descriptions;

   public:
    DAG(int N = 0) : n(N), In(N), Out(N) {}

    int size() const noexcept { return n; }

    void resize(int N) {
        cout << "Resizing DAG from " << n << " to " << N << endl;
        if (N <= n) return;
        In.resize(N);
        Out.resize(N);
        n = N;
    }

    void addDescriptionLine(const string& line) noexcept {
        // Split the line using the character '\n'
        stringstream ss(line);
        string token;
        while (getline(ss, token, '\n')) descriptions.push_back(token);
    }

    void addEdge(int v1, int v2, const string& description = "",
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
        // if (v1 > v2) {
        //     cerr << "DAG edge addition error: " << v1 << " > " << v2 << endl;
        //     abort();
        // }

        if (v1 >= Out.size() || v2 >= In.size()) {
            cerr << "DAG edge addition error: node index out of range." << endl;
            cerr << "v1: " << v1 << ", v2: " << v2 << endl;
            cerr << "n: " << n << ", Out.size(): " << Out.size()
                 << ", In.size(): " << In.size() << endl;
            abort();
        }

        In[v2].push_back(v1);
        Out[v1].push_back(v2);
    }

    void printDAG(const string& filename) const noexcept {
        // Replace filename extension by .dag.mtx (if any otherwise add it)
        string filename_ext = filename.substr(filename.find_last_of(".") + 1);
        string filename_noext = filename.substr(0, filename.find_last_of("."));
        string filename_dag = filename_noext + ".dag.mtx";

        size_t nnz = std::accumulate(
            Out.cbegin(), Out.cend(), 0,
            [](int sum, const vector<int>& v) { return sum + v.size(); });

        {
            ofstream outfile(filename_dag, ios::out);
            outfile << "%%MatrixMarket matrix coordinate real general\n";
            outfile << "%\n";
            for (const auto& each : descriptions)
                outfile << "% " << each << "\n";
            outfile << "%\n";
            // Print MM header
            outfile << n << " " << n << " " << nnz << "\n";
            // Print edges (directed)
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < Out[i].size(); ++j) {
                    outfile << i << " " << Out[i][j] << " " << "1" << "\n";
                }
            }
        }
    }

    // Prints the hyperDAG corresponding to the DAG into a file
    void printHyperDAG(const string& filename) const noexcept {
        int sinks = 0, pins = 0;
        for (int i = 0; i < n; ++i) {
            if (Out[i].empty()) {
                ++sinks;
            } else {
                pins += 1 + Out[i].size();
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
        int edgeIndex = 0;
        for (int i = 0; i < n; ++i) {
            if (Out[i].empty()) continue;

            outfile << edgeIndex << " " << 1 << "\n";
            ++edgeIndex;
        }

        // Print work weights of nodes - this is indegree-1
        outfile << "% Nodes ( id, work cost ):\n";
        for (int i = 0; i < n; ++i) {
            if (Out[i].empty() && In[i].empty()) continue;

            outfile << i << " " << max((int)In[i].size() - 1, 0) << "\n";
        }

        // Print all pins
        outfile << "% Pins ( hyperdedge.id, node.id ):\n";
        edgeIndex = 0;
        for (int i = 0; i < n; ++i) {
            if (Out[i].empty()) {
                continue;
            }

            cout << "Out of node " << i << ": ";
            for (int j = 0; j < Out[i].size(); ++j) {
                cout << Out[i][j] << " ";
            }
            cout << endl;

            outfile << edgeIndex << " " << i << "\n";
            for (int j = 0; j < Out[i].size(); ++j)
                outfile << edgeIndex << " " << Out[i][j] << "\n";

            ++edgeIndex;
        }
        outfile.close();
    }

    // Checks for each node whether it is connected (in an undirected sense) to
    // the set of source nodes (using a BFS) (if the onlyBack flag is set, then
    // we only search for the predecessors of the given nodes)
    vector<bool> isReachable(const vector<int>& sources,
                             bool onlyBack = false) const noexcept {
        vector<bool> visited(n, false);
        list<int> next;

        // Mark the input nodes as reachable
        for (int i = 0; i < sources.size(); ++i) {
            visited[sources[i]] = true;
            next.push_back(sources[i]);
        }

        // Execute BFS
        while (!next.empty()) {
            int node = next.front();
            next.pop_front();

            for (int i = 0; i < In[node].size(); ++i)
                if (!visited[In[node][i]]) {
                    next.push_back(In[node][i]);
                    visited[In[node][i]] = true;
                }

            if (onlyBack) continue;

            for (int i = 0; i < Out[node].size(); ++i)
                if (!visited[Out[node][i]]) {
                    next.push_back(Out[node][i]);
                    visited[Out[node][i]] = true;
                }
        }

        return visited;
    }

    // Checks if the DAG is a single connected component
    bool isConnected() const noexcept {
        vector<int> sources(1, 0);
        vector<bool> reachable = isReachable(sources);

        for (int i = 0; i < n; ++i)
            if (!reachable[i]) return false;

        return true;
    }

    // Creates a smaller, 'cleaned' DAG, consisting of only the specified nodes
    DAG keepGivenNodes(const vector<bool>& keepNode) {
        int NrOfNodes = 0;
        vector<int> newIdx(n);
        for (int i = 0; i < n; ++i)
            if (keepNode[i]) {
                newIdx[i] = i;  // NrOfNodes;
                ++NrOfNodes;
            }

        DAG cleaned(n);
        for (int i = 0; i < descriptions.size(); ++i)
            cleaned.addDescriptionLine(descriptions[i]);

        for (int i = 0; i < n; ++i)
            if (keepNode[i])
                for (int j = 0; j < Out[i].size(); ++j)
                    if (keepNode[Out[i][j]])
                        cleaned.addEdge(newIdx[i], newIdx[Out[i][j]], "", true);

        if (DebugMode) {
            cout << "Only the following nodes are kept: ";
            for (int i = 0; i < keepNode.size(); ++i)
                if (keepNode[i]) cout << i << " ";
            cout << endl;
        }

        return (cleaned);
    }

    void printConnected() const noexcept {
        if (isConnected())
            cout << "The DAG is connected.\n";
        else
            cout << "The DAG is NOT connected.\n";
    }
};

struct IMatrix {
   private:
    const int _m, _n;
    const string _label = "";
    string _desc = "";

   protected:
    IMatrix(int M, int N, const string& label = "")
        : _m(M), _n(N), _label(label) {}

   public:
    int nrows() const noexcept { return _m; }
    int ncols() const noexcept { return _n; }
    int area() const noexcept { return nrows() * ncols(); }
    virtual int nnz() const noexcept = 0;

    void addDescription(const string& desc) noexcept { _desc += desc + "\n"; }
    std::string getDescription() const noexcept { return _desc; }
    std::string getLabel() const noexcept { return _label; }

    virtual bool at(int i, int j) const noexcept = 0;
    virtual void set(int i, int j, bool value) noexcept = 0;

    void setMainDiagonal() {
        for (int i = 0; i < nrows(); ++i) {
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
        for (int i = 0; i < nrows(); ++i) {
            for (int j = 0; j < ncols(); ++j) {
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

    void print(const string& label = "",
               std::ostream& os = std::cout) const noexcept {
        os << "Matrix " << label << ": " << nrows() << "x" << ncols() << ", "
           << nnz() << " nonzeros\n";
        os << "[\n";
        for (int i = 0; i < nrows(); ++i) {
            os << "  ";
            for (int j = 0; j < ncols(); ++j) {
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
                                   bool IndexedFromOne = false) {
        ifstream infile(filename);
        if (!infile.is_open()) {
            throw "Unable to find/open input matrix file.\n";
        }

        string line;
        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%') getline(infile, line);

        int M, N, NNZ;
        sscanf(line.c_str(), "%d %d %d", &N, &M, &NNZ);

        if (NNZ == 0) {
            throw "Incorrect input file format: only non-empty matrices are "
                    "accepted!\n";
        }

        MatrixType A(M, N, "Matrix from file <" + filename + ">");
        // read nonzeros
        int indexOffset = IndexedFromOne ? 1 : 0;
        for (int i = 0; i < NNZ; ++i) {
            if (infile.eof()) {
                throw "Incorrect input file format (file terminated too "
                        "early).\n";
            }

            int x, y;
            getline(infile, line);
            while (!infile.eof() && line.at(0) == '%') getline(infile, line);

            sscanf(line.c_str(), "%d %d", &x, &y);

            if (x < indexOffset || y < indexOffset || x >= N + indexOffset ||
                y >= N + indexOffset) {
                throw "Incorrect input file format (index out of range).\n";
            }

            A.set(x - indexOffset, y - indexOffset, true);
        }

        A.addDescription("Matrix A is read from input file " + filename + ".");
        infile.close();

        return A;
    }

    template <typename MatrixType>
    static MatrixType Identity(int M, int N) {
        MatrixType I(M, N);
        I.setMainDiagonal();
        return I;
    }
};

struct SquareMatrix : public IMatrix {
   private:
    int _nnz = 0;
    vector<vector<bool> > _cells;

   public:
    SquareMatrix(int N = 0, const string& label = "") : IMatrix(N, N, label) {
        _cells.resize(N, vector<bool>(N, false));

        addDescription("Matrix A is a square-matrix of size " + to_string(N) +
                       "x" + to_string(N) + ".");
    }

    SquareMatrix(int _, int N, const string& label) : SquareMatrix(N, label) {}

    SquareMatrix(const SquareMatrix& other) : IMatrix(other) {
        _nnz = other._nnz;
        _cells = other._cells;
    }

    SquareMatrix& operator=(const SquareMatrix& other) {
        return *this = SquareMatrix(other);
    }

    int nnz() const noexcept override { return _nnz; }

    bool at(int i, int j) const noexcept override { return _cells[i][j]; }

    void set(int i, int j, bool value) noexcept override {
        if (at(i, j) == value) return;
        _cells[i][j] = value;
        _nnz += value ? 1 : -1;
    }
};

struct LowerTriangularSquareMatrix : public SquareMatrix {
   public:
    LowerTriangularSquareMatrix(int N = 0, const string& label = "")
        : SquareMatrix(N, label) {
        addDescription("Matrix A is a lower triangular square-matrix of size " +
                       to_string(N) + "x" + to_string(N) + ".");
    }

    LowerTriangularSquareMatrix(int _, int N, const string& label)
        : SquareMatrix(N, label) {}

    LowerTriangularSquareMatrix(const LowerTriangularSquareMatrix& other)
        : SquareMatrix(other) {}

    LowerTriangularSquareMatrix& operator=(
        const LowerTriangularSquareMatrix& other) {
        return *this = LowerTriangularSquareMatrix(other);
    }

    bool at(int i, int j) const noexcept override {
        if (i < j) return false;
        return SquareMatrix::at(i, j);
    }

    void set(int i, int j, bool value) noexcept override {
        if (i < j) return;
        SquareMatrix::set(i, j, value);
    }
};

struct UpperTriangularSquareMatrix : public SquareMatrix {
   public:
    UpperTriangularSquareMatrix(int N = 0, const string& label = "")
        : SquareMatrix(N, label) {
        addDescription("Matrix A is a upper triangular square-matrix of size " +
                       to_string(N) + "x" + to_string(N) + ".");
    }

    UpperTriangularSquareMatrix(int _, int N, const string& label)
        : SquareMatrix(N, label) {}

    UpperTriangularSquareMatrix(const UpperTriangularSquareMatrix& other)
        : SquareMatrix(other) {}

    UpperTriangularSquareMatrix& operator=(
        const UpperTriangularSquareMatrix& other) {
        return *this = UpperTriangularSquareMatrix(other);
    }

    bool at(int i, int j) const noexcept override {
        if (i > j) return false;
        return SquareMatrix::at(i, j);
    }

    void set(int i, int j, bool value) noexcept override {
        if (i > j) return;
        SquareMatrix::set(i, j, value);
    }
};
DAG CreateRandomIndegExpected(int N, double indeg, double sourceProb = 0.0) {
    DAG G(N);

    G.addDescriptionLine("HyperDAG from a random DAG with expected indegree " +
                         to_string(indeg) + ".");
    G.addDescriptionLine("Each node is chosen as a source with probability " +
                         to_string(sourceProb) + ".");

    for (int i = 1; i < N; ++i) {
        if ((double)rand() / (double)RAND_MAX < sourceProb) continue;

        double p = indeg / (double)i;

        for (int j = 0; j < i; ++j)
            if ((double)rand() / (double)RAND_MAX < p) G.addEdge(j, i);
    }

    return G;
}

DAG CreateRandomIndegFixed(int N, int indeg, double sourceProb = 0.0) {
    DAG G(N);

    G.addDescriptionLine("HyperDAG from a random DAG with fixed indegree " +
                         to_string(indeg) + ".");
    G.addDescriptionLine("Each node is chosen as a source with probability " +
                         to_string(sourceProb) + ".");

    for (int i = 1; i < N; ++i) {
        if ((double)rand() / (double)RAND_MAX < sourceProb) continue;

        if (i <= indeg)  // all previous nodes are chosen
            for (int j = 0; j < i; ++j) G.addEdge(j, i);

        else  // chose 'indeg' predecessors at random
        {
            vector<bool> chosen(i, false);
            for (int j = 0; j < indeg; ++j) {
                int rnd = randInt(i - j), idx = 0;
                for (; chosen[idx]; ++idx)
                    ;
                for (int k = 0; k < rnd; ++k) {
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

DAG CreateRandomER(int N, int NrOfEdges) {
    DAG G(N);

    G.addDescriptionLine(
        "HyperDAG from a random DAG with expected number of edges " +
        to_string(NrOfEdges) +
        ", with uniform edge probabilities on all forward edges");

    double p = 2.0 * (double)NrOfEdges / ((double)N * ((double)N - 1));

    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
            if ((double)rand() / (double)RAND_MAX < p) G.addEdge(i, j);

    return G;
}

/**
 * @brief Complete a DAG with an out-of-place SpMV operation.
 *
 * @param hyperdag DAG to extend, may be empty.
 * @param M        Matrix to multiply with vector v, which is
 *                 assumed to be of size M.n and dense.
 */
void CreateSpMV(DAG& hyperdag, const SquareMatrix& M) {
    if (DebugMode) M.print("SpMV");

    const int N = M.nrows();

    // Offsets
    const int offset = hyperdag.size();
    const int MOffset = offset;
    const int vOffset = MOffset + N * N;
    const int mulMvOffset = vOffset + N;
    const int uOffset = mulMvOffset + N * N;

    // Create hyperDAG locally
    DAG G = hyperdag;
    const int nNodes = uOffset + N;
    G.resize(hyperdag.size() + nNodes);
    G.addDescriptionLine("HyperDAG model of SpMV operation.");

    // find empty rows in the matrix
    vector<bool> rowNotEmpty(N, false);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (M.at(i, j)) rowNotEmpty[i] = true;

    // create SpMV DAG
    for (int i = 0; i < N; ++i) {
        if (!rowNotEmpty[i]) continue;

        for (int j = 0; j < N; ++j) {
            if (not M.at(i, j)) continue;

            // // Operation: u[j] += M[i][j] * v[i]
            // Nodes
            int node_M_i_j = MOffset + i * N + j;
            int node_v_i = vOffset + i;
            int node_mul_M_v = mulMvOffset + i * N + j;
            int node_u_j = uOffset + j;
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
    vector<int> sinkNodes;
    for (int i = 0; i < N; ++i)
        if (rowNotEmpty[i]) sinkNodes.push_back(uOffset + i);

    hyperdag = G.keepGivenNodes(G.isReachable(sinkNodes));
}

DAG CreateRandomSpMV(int N, double nonzero) {
    SquareMatrix M(N);
    M.randomize(nonzero);

    DAG G;
    CreateSpMV(G, M);
    return G;
}

void CreateLLtSolver(DAG& hyperdag, const SquareMatrix& L) {
    if (DebugMode) L.print("L");

    const int n = L.nrows();
    const int nSquared = n * n;

    // Offsets
    const int offset = hyperdag.size();
    const int LOffset = offset;
    const int xOffset = LOffset + nSquared;
    const int y_MulOffset = xOffset + n;
    const int y_SubOffset = y_MulOffset + nSquared;
    const int yOffset = y_SubOffset + n;
    const int z_MulOffset = yOffset + n;
    const int z_SubOffset = z_MulOffset + nSquared;
    const int zOffset = z_SubOffset + n;

    const int nNodes = zOffset + n;
    cout << "nNodes: " << nNodes << endl;
    DAG G = hyperdag;
    G.resize(hyperdag.size() + nNodes);

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

    // find empty rows in the matrix L
    vector<bool> L_rowNotEmpty(n, false);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            if (L.at(i, j)) {
                L_rowNotEmpty[i] = true;
                break;
            }
    }

    // Forward substitution DAG
    for (int i = 0; i < n; ++i) {
        if (!L_rowNotEmpty[i]) continue;
        if (DebugMode) cout << "-- row " << i << ":\n";

        const int y_i = yOffset + i;
        const string y_i_str = "y[" + to_string(i) + "]";

        const int x_i = xOffset + i;
        const string x_i_str = "x[" + to_string(i) + "]";

        const int L_i_i = LOffset + i * n + i;
        const string L_i_i_str =
            "L[" + to_string(i) + "][" + to_string(i) + "]";

        const int sub_x_Mul = y_SubOffset + i;
        const string sub_x_Mul_str =
            "(" + x_i_str + " - sum(y[j] * L[" + to_string(i) + "][j]))";

        for (int j = 0; j < i; ++j) {
            if (not L.at(i, j)) continue;

            const int L_i_j = LOffset + i * n + j;
            const string L_i_j_str =
                "L[" + to_string(i) + "][" + to_string(j) + "]";

            const int y_j = yOffset + j;
            const string y_j_str = "y[" + to_string(j) + "]";

            const int mul_L_x = y_MulOffset + i * n + j;
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
    vector<bool> L_colNotEmpty(n, false);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (L.at(i, j)) {
                L_colNotEmpty[j] = true;
                break;
            }
        }
    }

    // Backward substitution DAG
    for (int i = n - 1; i >= 0; --i) {
        if (!L_colNotEmpty[i]) continue;
        if (DebugMode) cout << "-- row " << i << ":\n";

        const int z_i = zOffset + i;
        const string z_i_str = "z[" + to_string(i) + "]";

        const int y_i = yOffset + i;
        const string y_i_str = "y[" + to_string(i) + "]";

        const int Lt_i_i = LOffset + i * n + i;
        const string Lt_i_i_str =
            "Lt[" + to_string(i) + "][" + to_string(i) + "]";

        const int sub_x_Mul = z_SubOffset + i;
        const string sub_x_Mul_str =
            "(" + y_i_str + " - sum(z[j] * Lt[" + to_string(i) + "][j]))";

        for (int j = n - 1; j > i; --j) {
            if (not L.at(j, i)) continue;

            const int Lt_i_j = LOffset + j * n + i;
            const string Lt_i_j_str =
                "Lt[" + to_string(i) + "][" + to_string(j) + "]";

            const int z_j = zOffset + j;
            const string z_j_str = "z[" + to_string(j) + "]";

            const int mul_L_x = z_MulOffset + i * n + j;
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

    // only keep components that are predecessors of the final nonzeros
    // (in particular, indeces corresponding to (i) empty columns in the
    // original vector or (ii) emtpy rows in the result vector)
    // vector<int> sinkNodes;
    // for (int i = 0; i < n; ++i)
    //     if (L_rowNotEmpty[i] || L_colNotEmpty[i])
    //         sinkNodes.push_back(zOffset + i);

    // hyperdag = G.keepGivenNodes(G.isReachable(sinkNodes));

    hyperdag = G;
}

DAG CreateRandomLLtSolver(int N, double nonzero) {
    LowerTriangularSquareMatrix L(N);
    L.randomize(nonzero);
    DAG G;
    CreateLLtSolver(G, L);
    return G;
}

void CreateALLtSolver(DAG& G, const LowerTriangularSquareMatrix& L,
                      const SquareMatrix& A) {
    throw std::runtime_error("Not implemented: ALLtSolver");
    if (DebugMode) L.print("ALLtSolver");
    return CreateLLtSolver(G, L);
}

DAG CreateRandomALLtSolver(int N, double nonzero) {
    LowerTriangularSquareMatrix L(N);
    L.randomize(nonzero);
    SquareMatrix A(N);
    A.randomize(nonzero);
    DAG G;
    CreateALLtSolver(G, L, A);
    return G;
}

void CreateLUSolver(DAG& hyperdag, const LowerTriangularSquareMatrix& L,
                    const UpperTriangularSquareMatrix& U) {
    if (DebugMode) {
        L.print("LUSolver - L");
        U.print("LUSolver - U");
    }

    if (L.nrows() != U.nrows()) {
        cerr << "Error: L and U matrices must have the same size." << endl;
        abort();
    }

    const int n = L.nrows();
    const int nSquared = n * n;

    // Offsets
    const int offset = hyperdag.size();
    const int LOffset = offset;
    const int UOffset = LOffset + nSquared;
    const int xOffset = UOffset + nSquared;
    const int y_MulOffset = xOffset + n;
    const int y_SubOffset = y_MulOffset + nSquared;
    const int yOffset = y_SubOffset + n;
    const int z_MulOffset = yOffset + n;
    const int z_SubOffset = z_MulOffset + nSquared;
    const int zOffset = z_SubOffset + n;

    const int nNodes = zOffset + n;
    cout << "nNodes: " << nNodes << endl;
    DAG G = hyperdag;
    G.resize(hyperdag.size() + nNodes);

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

    // find empty rows in the matrix L
    vector<bool> L_rowNotEmpty(n, false);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            if (L.at(i, j)) {
                L_rowNotEmpty[i] = true;
                break;
            }
    }

    // Forward substitution DAG
    for (int i = 0; i < n; ++i) {
        if (!L_rowNotEmpty[i]) continue;
        if (DebugMode) cout << "-- row " << i << ":\n";

        const int y_i = yOffset + i;
        const string y_i_str = "y[" + to_string(i) + "]";

        const int x_i = xOffset + i;
        const string x_i_str = "x[" + to_string(i) + "]";

        const int L_i_i = LOffset + i * n + i;
        const string L_i_i_str =
            "L[" + to_string(i) + "][" + to_string(i) + "]";

        const int sub_x_Mul = y_SubOffset + i;
        const string sub_x_Mul_str =
            "(" + x_i_str + " - sum(y[j] * L[" + to_string(i) + "][j]))";

        for (int j = 0; j < i; ++j) {
            if (not L.at(i, j)) continue;

            const int L_i_j = LOffset + i * n + j;
            const string L_i_j_str =
                "L[" + to_string(i) + "][" + to_string(j) + "]";

            const int y_j = yOffset + j;
            const string y_j_str = "y[" + to_string(j) + "]";

            const int mul_L_x = y_MulOffset + i * n + j;
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
    vector<bool> U_rowNotEmpty(n, false);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (U.at(i, j)) {
                U_rowNotEmpty[j] = true;
                break;
            }
        }
    }

    // Backward substitution DAG
    for (int i = n - 1; i >= 0; --i) {
        if (!U_rowNotEmpty[i]) continue;
        if (DebugMode) cout << "-- row " << i << ":\n";

        const int z_i = zOffset + i;
        const string z_i_str = "z[" + to_string(i) + "]";

        const int y_i = yOffset + i;
        const string y_i_str = "y[" + to_string(i) + "]";

        const int U_i_i = UOffset + i * n + i;
        const string U_i_i_str =
            "U[" + to_string(i) + "][" + to_string(i) + "]";

        const int sub_x_Mul = z_SubOffset + i;
        const string sub_x_Mul_str =
            "(" + y_i_str + " - sum(z[j] * U[" + to_string(i) + "][j]))";

        for (int j = n - 1; j > i; --j) {
            if (not U.at(i, j)) continue;

            const int U_i_j = UOffset + i * n + j;
            const string U_i_j_str =
                "U[" + to_string(i) + "][" + to_string(j) + "]";

            const int z_j = zOffset + j;
            const string z_j_str = "z[" + to_string(j) + "]";

            const int mul_U_x = z_MulOffset + i * n + j;
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

    // only keep components that are predecessors of the final nonzeros
    // (in particular, indeces corresponding to (i) empty columns in the
    // original vector or (ii) emtpy rows in the result vector)
    vector<int> sinkNodes(n);
    cout << "sinkNodes: ";
    for (int i = 0; i < n; ++i) {
        sinkNodes[i] = zOffset + i;
        cout << sinkNodes[i] << " ";
    }
    cout << endl;

    hyperdag = G;  //.keepGivenNodes(G.isReachable(sinkNodes));
}

DAG CreateRandomLUSolver(int N, double nonzero) {
    LowerTriangularSquareMatrix L(N);
    UpperTriangularSquareMatrix U(N);
    L.randomize(nonzero);
    U.randomize(nonzero);
    DAG G;
    CreateLUSolver(G, L, U);
    return G;
}

DAG CreateSpMVExp(const SquareMatrix& M, int K) {
    int N = M.nrows();

    DAG G((K + 1) * (M.nnz() + N));
    G.addDescriptionLine("HyperDAG model of naive implementation of A^" +
                         to_string(K) +
                         " *v with sparse matrix A and dense vector v.");
    G.addDescriptionLine(M.getDescription());

    // Index of the nodes representing each nonzero of M
    vector<bool> rowValid(N, true), colValid(N, true);
    vector<vector<int> > originalCellIdx(N, vector<int>(N));
    int Idx = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (M.at(i, j)) {
                originalCellIdx[i][j] = Idx;
                ++Idx;
            }
        }
    }

    // Vbase is the starting point of the current vector v, base is the start
    // index for the next iteration
    int Vbase = M.nnz(), base = Vbase + N;

    // ITERATIONS
    for (int k = 0; k < K; ++k) {
        colValid = rowValid;

        // count the number of used cells in current iteration
        int usedCells = 0;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
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
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
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
    vector<int> finalVector;
    for (int i = 0; i < N; ++i)
        if (rowValid[i]) finalVector.push_back(Vbase + i);

    return G.keepGivenNodes(G.isReachable(finalVector, true));
}

DAG CreateRandomSpMVExp(int N, double nonzero, int K) {
    SquareMatrix M(N);
    M.randomize(nonzero);

    DAG G = CreateSpMVExp(M, K);

    return G;
}

DAG CreateSparseCG(const SquareMatrix& M, int K) {
    int N = M.nrows();

    DAG G(3 * N - 1 + 2 * M.nnz() + K * (M.nnz() + 9 * N + 4));
    G.addDescriptionLine("HyperDAG model of naive implementation of " +
                         to_string(K) +
                         " iterations of the conjugate gradient method.");
    G.addDescriptionLine(M.getDescription());

    // Index of the nodes representing each nonzero of M
    // (we will denote the matrix M by "A" in the comment pseudocodes, since it
    // is more standard)
    vector<vector<int> > originalCellIdx(N, vector<int>(N));
    int Idx = 0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (M.at(i, j)) {
                originalCellIdx[i][j] = Idx;
                ++Idx;
            }

    // INITIALIZATION PHASE

    // compute A * x_0
    Idx = 0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            if (M.at(i, j)) {
                G.addEdge(M.nnz() + j, M.nnz() + N + Idx);
                G.addEdge(originalCellIdx[i][j], M.nnz() + N + Idx);
                G.addEdge(M.nnz() + N + Idx, 2 * M.nnz() + N + i);
                Idx += 1;
            }
        }

    // Rbase, Pbase and Xbase points to the beginning of the current r, p and x
    // vectors
    int Rbase = 2 * M.nnz() + 3 * N, Pbase = Rbase + N, Xbase = M.nnz();

    // r_0 := b - A * x_0 ;  p_0 := r_0
    for (int i = 0; i < N; ++i) {
        G.addEdge(2 * M.nnz() + N + i, Rbase + i);
        G.addEdge(2 * M.nnz() + 2 * N + i, Rbase + i);
        G.addEdge(Rbase + i, Pbase + i);
    }

    // compute r_0^T * r_0 and save its index in Rproduct
    int Rproduct = Pbase + 2 * N;
    for (int i = 0; i < N; ++i) {
        G.addEdge(Rbase + i, Pbase + N + i);
        G.addEdge(Pbase + N + i, Rproduct);
    }

    // ITERATIONS

    // base denotes the current index at the beginning of each iteration
    int base = Rproduct + 1;

    for (int k = 0; k < K; ++k) {
        // compute A * p_k
        Idx = 0;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                if (M.at(i, j)) {
                    G.addEdge(Pbase + j, base + Idx);
                    G.addEdge(originalCellIdx[i][j], base + Idx);
                    G.addEdge(base + Idx, base + M.nnz() + i);
                    Idx += 1;
                }
            }

        // compute p_k^T * A * p_k
        for (int i = 0; i < N; ++i) {
            G.addEdge(Pbase + i, base + M.nnz() + N + i);
            G.addEdge(base + M.nnz() + i, base + M.nnz() + N + i);
            G.addEdge(base + M.nnz() + N + i, base + M.nnz() + 2 * N);
        }

        // alpha_k := r_k^T * r_k / p_k^T * A * p_k
        G.addEdge(base + M.nnz() + 2 * N, base + M.nnz() + 2 * N + 1);
        G.addEdge(Rproduct, base + M.nnz() + 2 * N + 1);

        // index of alpha_k
        int alpha = base + M.nnz() + 2 * N + 1;

        // x_(k+1) := x_k + alpha_k * p_k
        for (int i = 0; i < N; ++i) {
            G.addEdge(Pbase + i, alpha + 1 + i);
            G.addEdge(alpha, alpha + 1 + i);
            G.addEdge(alpha + 1 + i, alpha + 1 + N + i);
            G.addEdge(Xbase + i, alpha + 1 + N + i);
        }
        Xbase = alpha + 1 + N;

        // r_(k+1) := r_k - alpha_k * A * p_k
        for (int i = 0; i < N; ++i) {
            G.addEdge(base + M.nnz() + i, Xbase + N + i);
            G.addEdge(alpha, Xbase + N + i);
            G.addEdge(Xbase + N + i, Xbase + 2 * N + i);
            G.addEdge(Rbase + i, Xbase + 2 * N + i);
        }
        Rbase = Xbase + 2 * N;

        if (k == K - 1) break;

        // compute r_(k+1)^T * r_(k+1)
        for (int i = 0; i < N; ++i) {
            G.addEdge(Rbase + i, Rbase + N + i);
            G.addEdge(Rbase + N + i, Rbase + 2 * N);
        }

        // beta_k := r_(k+1)^T * r_(k+1) / r_k^T * r_k
        G.addEdge(Rbase + 2 * N, Rbase + 2 * N + 1);
        G.addEdge(Rproduct, Rbase + 2 * N + 1);

        // updated index of Rproduct and beta_k
        Rproduct = Rbase + 2 * N;
        int beta = Rbase + 2 * N + 1;

        // p_(k+1) := r_(k+1) - beta_k * p_k
        for (int i = 0; i < N; ++i) {
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

DAG CreateRandomSparseCG(int N, double nonzero, int K) {
    SquareMatrix M(N);
    M.randomize(nonzero);

    DAG G = CreateSparseCG(M, K);

    return G;
}

DAG CreatekNN(const SquareMatrix& M, int K, int source) {
    int N = M.nrows();

    DAG G((K + 1) * (M.nnz() + N));
    G.addDescriptionLine(
        "HyperDAG model of naive implementation of " + to_string(K) +
        " iterations of kNN, starting from node number " + to_string(source) +
        " (i.e. sparse vector with 1 entry).");
    G.addDescriptionLine(M.getDescription());

    // Index of the nodes representing each nonzero of M
    vector<vector<int> > originalCellIdx(N, vector<int>(N));
    int Idx = 0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (M.at(i, j)) {
                originalCellIdx[i][j] = Idx;
                ++Idx;
            }

    // Initialize sparse vector
    vector<bool> reached(N, false);
    reached[source] = true;

    // Vbase is the current index of the beginning of the vector
    int Vbase = Idx;
    // base is the index where the next iteration starts
    int base = Idx + 1;

    // current new index for each original row of the sparse vector
    vector<int> newRowIdx(N);
    newRowIdx[source] = 0;

    // Iterations
    for (int k = 0; k < K; ++k) {
        // count the number of internal nodes (matrix-vector multiplication) in
        // this iteration
        int NrOfInternalNodes = 0;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (reached[j] && M.at(i, j)) ++NrOfInternalNodes;

        // add DAG edges
        int rowIdx = 0, cellIdx = 0;
        vector<bool> rowNotEmpty(N, false);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j)
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
        for (int i = 0; i < N; ++i)
            if (rowNotEmpty[i]) {
                reached[i] = true;
                newRowIdx[i] = rowIdx;
                ++rowIdx;
            }

        int oldbase = base;
        base = oldbase + NrOfInternalNodes + rowIdx;
        Vbase = oldbase + NrOfInternalNodes;
    }

    // only keep components that are connected to the final nonzeros
    // (in particular, this removes nonzeros of the matrix that were never used)
    vector<int> finalVector;
    for (int i = Vbase; i < base; ++i) finalVector.push_back(i);

    return G.keepGivenNodes(G.isReachable(finalVector));
}

DAG CreateRandomkNN(int N, double nonzero, int K) {
    SquareMatrix M(N);
    M.randomize(nonzero);
    M.setMainDiagonal();

    int source = randInt(N);

    DAG G = CreatekNN(M, K, source);

    return G;
}

int main(int argc, char* argv[]) {
    int N = -1, edges = -1, K = -1, sourceNode = -1, indegFix = -1;
    double sourceProb = -1.0, nonzeroProb = -1.0, indegExp = -1.0;
    string infile, outfile, mode, indegreeString;
    bool indexedFromOne = false;

    // PROCESS COMMAND LINE ARGUMENTS

    vector<string> params{
        "-output", "-input",    "-mode",       "-N",           "-K",
        "-edges",  "-indegree", "-sourceProb", "-nonzeroProb", "-sourceNode"};
    vector<string> modes{"ER",
                         "fixedIn",
                         "expectedIn",
                         "SpMV",
                         "SpMVExp",
                         "ALLtSolver",
                         "ALLtSolverExp",
                         "LLtSolver",
                         "LLtSolverExp",
                         "LUSolver",
                         "LUSolverExp",
                         "kNN",
                         "CG"};

    for (int i = 1; i < argc; ++i) {
        // Check parameters that require an argument afterwards
        if (contains(params, string(argv[i])) && i + 1 >= argc) {
            cerr << "Parameter error: no parameter value after the \""
                 << string(argv[i]) << "\" option." << endl;
            return 1;
        }

        if (string(argv[i]) == "-output")
            outfile = argv[++i];

        else if (string(argv[i]) == "-input")
            infile = argv[++i];

        else if (string(argv[i]) == "-mode") {
            mode = argv[++i];
            if (!contains(modes, mode)) {
                cerr << "Parameter error: invalid mode parameter. Please check "
                        "the readme for possible modes."
                     << endl;
                return 1;
            }
        }

        else if (string(argv[i]) == "-N") {
            N = stoi(argv[++i]);
            if (N <= 1) {
                cerr << "Parameter error: N has to be at least 2." << endl;
                return 1;
            }
        }

        else if (string(argv[i]) == "-K") {
            K = stoi(argv[++i]);
            if (K <= 0) {
                cerr << "Parameter error: K has to be at least 1." << endl;
                return 1;
            }
        }

        else if (string(argv[i]) == "-edges")
            edges = stoi(argv[++i]);

        else if (string(argv[i]) == "-indegree")
            indegreeString = argv[++i];

        else if (string(argv[i]) == "-sourceProb") {
            sourceProb = stod(argv[++i]);
            if (sourceProb < -0.0001 || sourceProb > 1.0001) {
                cerr << "Parameter error: parameter \"sourceProb\" has to be "
                        "between 0 and 1."
                     << endl;
                return 1;
            }
        }

        else if (string(argv[i]) == "-nonzeroProb") {
            nonzeroProb = stod(argv[++i]);
            if (nonzeroProb < -0.0001 || nonzeroProb > 1.0001) {
                cerr << "Parameter error: parameter \"nonzeroProb\" has to be "
                        "between 0 and 1."
                     << endl;
                return 1;
            }
        }

        else if (string(argv[i]) == "-sourceNode") {
            sourceNode = stoi(argv[++i]);
            if (sourceNode < 0) {
                cerr << "Parameter error: sourceNode cannot be negative."
                     << endl;
                return 1;
            }
        }

        else if (string(argv[i]) == "-debugMode")
            DebugMode = true;

        else if (string(argv[i]) == "-indexedFromOne")
            indexedFromOne = true;

        else {
            cerr << "Parameter error: unknown parameter/option "
                 << string(argv[i]) << endl;
            return 1;
        }
    }

    // CHECK PARAMETER CONSISTENCY

    // mode, infile, outfile
    if (mode.empty()) {
        cerr << "Parameter error: no mode specified." << endl;
        return 1;
    }
    if (!infile.empty() &&
        (mode == "ER" || mode == "fixedIn" || mode == "expectedIn")) {
        cerr << "Parameter error: cannot use input matrix file for this mode."
             << endl;
        return 1;
    }
    if (outfile.empty()) {
        outfile = "output.txt";
        if (DebugMode)
            cout << "Output file not specified; using default output filename "
                    "(output.txt)."
                 << endl;
    }
    if (infile.empty() && indexedFromOne) {
        cerr << "Parameter error: cannot use parameter \"indexedFromOne\" "
                "without an input file."
             << endl;
        return 1;
    }

    // N
    if (!infile.empty()) {
        if (N != -1) {
            cerr << "Parameter error: cannot use parameter \"N\" if input file "
                    "is specified."
                 << endl;
            return 1;
        }
    }
    // If not set, use 10 as default value
    else if (N == -1)
        N = 10;

    // K
    if (mode == "ER" || mode == "fixedIn" || mode == "expectedIn" ||
        mode == "SpMV" || mode == "LLtSolver" || mode == "ALLtSolver" ||
        mode == "LUSolver") {
        if (K >= 0) {
            cerr << "Parameter error: cannot use parameter \"K\" for this mode."
                 << endl;
            return 1;
        }
    } else {
        // If not set, use 3 as default value
        if (K < 0) K = 3;
    }

    // edges
    if (mode != "ER") {
        if (edges >= 0) {
            cerr << "Parameter error: cannot use parameter \"edges\" for this "
                    "mode."
                 << endl;
            return 1;
        }
    }
    // If not set, use 5*N as default value
    else if (edges < 0)
        edges = N * (N - 1) / 5;

    // indegree
    if (mode != "fixedIn" && mode != "expectedIn") {
        if (!indegreeString.empty()) {
            cerr << "Parameter error: cannot use parameter \"indegree\" for "
                    "this mode."
                 << endl;
            return 1;
        }
    } else if (mode == "fixedIn") {
        // If not set, use 4 as default value
        if (indegreeString.empty())
            indegFix = 4;
        else
            indegFix = stoi(indegreeString);

        if (indegFix <= 0 || indegFix >= N) {
            cerr << "Parameter error: parameter \"indegree\" has to be between "
                    "1 and (N-1)."
                 << endl;
            return 1;
        }
    } else if (mode == "expectedIn") {
        // If not set, use 4 as default value
        if (indegreeString.empty())
            indegExp = 4.0;
        else
            indegExp = stod(indegreeString);

        if (indegExp <= -0.0001 || indegExp >= N - 0.0009) {
            cerr << "Parameter error: parameter \"indegree\" has to be between "
                    "0 and (N-1)."
                 << endl;
            return 1;
        }
    }

    // sourceProb
    if (mode != "fixedIn" && mode != "expectedIn") {
        if (sourceProb > -0.5) {
            cerr << "Parameter error: cannot use \"sourceProb\" parameter for "
                    "this mode."
                 << endl;
            return 1;
        }
    }
    // If not set, use 0 as default value
    else if (sourceProb < -0.5)
        sourceProb = 0.0;

    // nonzeroProb
    if (mode == "ER" || mode == "fixedIn" || mode == "expectedIn") {
        if (nonzeroProb > -0.5) {
            cerr << "Parameter error: cannot use parameter \"nonzeroProb\" for "
                    "this mode."
                 << endl;
            return 1;
        }
    } else {
        if (nonzeroProb > -0.5 && !infile.empty()) {
            cerr << "Parameter error: parameter \"nonzeroProb\" has no use if "
                    "input file is specified."
                 << endl;
            return 1;
        }
        // If not set, use 0.25 as default value
        if (nonzeroProb < -0.5) nonzeroProb = 0.25;
    }

    // sourceNode
    if (sourceNode >= 0 && mode != "kNN") {
        cerr << "Parameter error: \"sourceNode\" parameter can only be used in "
                "kNN mode."
             << endl;
        return 1;
    } else if (sourceNode >= 0 && infile.empty()) {
        cerr << "Parameter error: \"sourceNode\" parameter is only used when "
                "the matrix is read from a file."
             << endl;
        return 1;
    }
    // Use 0 as default value
    else if (mode == "kNN" && sourceNode == -1)
        sourceNode = 0;

    // INITIALIZE
    srand(0);

    SquareMatrix M;
    if (!infile.empty()) {
        M = IMatrix::readFromFile<SquareMatrix>(infile, indexedFromOne);
        N = M.nrows();
    }

    // last parameter check (has to be done after N is read from file)
    if (mode == "kNN" && sourceNode >= N) {
        cout << "Parameter error: sourceNode is a node index, it cannot be "
                "larger than (N-1)."
             << endl;
        return 1;
    }

    // GENERATE HYPERDAG

    DAG G(1);

    if (mode == "ER") {
        G = CreateRandomER(N, edges);
    } else if (mode == "fixedIn") {
        G = CreateRandomIndegFixed(N, indegFix, sourceProb);
    } else if (mode == "expectedIn") {
        G = CreateRandomIndegExpected(N, indegExp, sourceProb);
    } else if (mode == "SpMV") {
        if (!infile.empty())
            CreateSpMV(G, M);
        else
            G = CreateRandomSpMV(N, nonzeroProb);
    } else if (mode == "SpMVExp") {
        if (!infile.empty())
            G = CreateSpMVExp(M, K);
        else
            G = CreateRandomSpMVExp(N, nonzeroProb, K);
    } else if (mode == "LLtSolver") {
        if (!infile.empty())
            CreateLLtSolver(G, M);
        else
            G = CreateRandomLLtSolver(N, nonzeroProb);
    } else if (mode == "LLtSolverExp") {
        // Not supported yet
        throw std::runtime_error("LLtSolverExp not supported yet.");
        // if(!infile.empty())
        //     G = CreateLLtSolverExp(M, K);
        // else
        //     G = CreateRandomLLtSolverExp(N, nonzeroProb, K);
    } else if (mode == "ALLtSolver") {
        if (!infile.empty())
            throw runtime_error("ALLtSolver not supported for input file.");
        else
            G = CreateRandomALLtSolver(N, nonzeroProb);
    } else if (mode == "ALLtSolverExp") {
        // Not supported yet
        throw std::runtime_error("ALLtSolverExp not supported yet.");
        // if(!infile.empty())
        //     G = CreateLLtSolverExp(M, K);
        // else
        //     G = CreateRandomLLtSolverExp(N, nonzeroProb, K);
    } else if (mode == "LUSolver") {
        if (!infile.empty())
            throw runtime_error("LUSolver not supported for input file.");
        else
            G = CreateRandomLUSolver(N, nonzeroProb);
    } else if (mode == "LUSolverExp") {
        // Not supported yet
        throw std::runtime_error("LUSolverExp not supported yet.");
        // if(!infile.empty())
        //     G = CreateLUSolverExp(M, K);
        // else
        //     G = CreateRandomLUSolverExp(N, nonzeroProb, K);
    } else if (mode == "kNN") {
        if (!infile.empty()) {
            M.setMainDiagonal();
            G = CreatekNN(M, K, sourceNode);
        } else
            G = CreateRandomkNN(N, nonzeroProb, K);
    } else if (mode == "CG") {
        if (!infile.empty())
            G = CreateSparseCG(M, K);
        else
            G = CreateRandomSparseCG(N, nonzeroProb, K);
    }

    if (DebugMode) G.printConnected();

    G.printDAG(outfile);
    G.printHyperDAG(outfile);
    return 0;
}
