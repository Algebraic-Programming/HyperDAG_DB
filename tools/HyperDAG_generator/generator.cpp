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


#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <list>
#include <algorithm>


using namespace std;

bool DebugMode = true;

// AUXILIARY FUNCTIONS

// unbiased random int generator
int randInt(int lim)
{
    int rnd = rand();
    while(rnd>=RAND_MAX-RAND_MAX%lim)
        rnd = rand();

    return rnd%lim;
}

// check if vector contains a specific element
bool contains(const vector<string>& vec, const string& s)
{
    for(int i=0; i<vec.size(); ++i)
        if(vec[i]==s)
            return true;

    return false;
}


// MAIN DATA STRUCTURES

struct DAG
{
    int n;
    vector< vector <int> > In, Out;
    string desc;

    DAG(int N)
    {
        n=N;
        In.resize(n);
        Out.resize(n);
    }

    void addEdge(int v1, int v2, const std::string& description = "", bool noPrint = false)
    {
        if(v1>=v2){
            cerr<<"DAG edge addition error."<<endl;
            abort();
        }

        if( v1 >= Out.size() || v2 >= In.size() ){
            cerr<<"DAG edge addition error: node index out of range."<<endl;
            abort();
        }

        if(DebugMode && !noPrint)
            cout<<"Edge ( "<<v1<<", "<<v2<<" ){ "<<description<<" }\n" << flush;
        
        In[v2].push_back(v1);
        Out[v1].push_back(v2);
    }

    // Prints the hyperDAG corresponding to the DAG into a file
    void printHyperDAG(string filename)
    {
        int sinks=0, pins=0;
        for(int i=0; i<n; ++i)
            if(Out[i].size()>0)
                pins+=1+Out[i].size();
            else
                ++sinks;

        ofstream outfile;
        outfile.open (filename);
        outfile << "%" << desc <<"\n";
        outfile << "% Hyperdeges: " << n - sinks <<"\n";
        outfile << "% Nodes: " << n <<"\n";
        outfile << "% Pins: " << pins <<"\n";
        outfile << n - sinks << " "<< n << " " << pins <<"\n";

        // Print communication weights of hyperedges - uniformly 1
        outfile << "% Hyperedges ( id, communication cost ):\n";
        int edgeIndex = 0;
        for(int i=0; i<n; ++i)
            if(Out[i].size()>0)
            {
                outfile << edgeIndex << " " << 1 <<"\n";
                ++edgeIndex;
            }

        // Print work weights of nodes - this is indegree-1
        outfile << "% Nodes ( id, work cost ):\n";
        for(int i=0; i<n; ++i)
            outfile << i << " " << max((int)In[i].size()-1, 0) <<"\n";

        // Print all pins
        outfile << "% Pins ( hyperdedge.id, node.id ):\n";
        edgeIndex = 0;
        for(int i=0; i<n; ++i)
            if(Out[i].size()>0)
            {
                outfile << edgeIndex << " "<< i <<"\n";
                for(int j=0; j<Out[i].size(); ++j)
                    outfile << edgeIndex << " "<< Out[i][j] <<"\n";

                ++edgeIndex;
            }
        outfile.close();
     }

     // Checks for each node whether it is connected (in an undirected sense) to the set of source nodes (using a BFS)
     // (if the onlyBack flag is set, then we only search for the predecessors of the given nodes)
     vector<bool> isReachable(const vector<int>& sources, bool onlyBack = false)
     {
         vector<bool> visited(n, false);
         list<int> next;

         // Mark the input nodes as reachable
         for(int i=0; i<sources.size(); ++i)
         {
            visited[sources[i]]=true;
            next.push_back(sources[i]);
         }

         // Execute BFS
         while(!next.empty())
         {

            int node = next.front();
            next.pop_front();

            for(int i=0; i<In[node].size(); ++i)
                if(!visited[In[node][i]])
                {
                    next.push_back(In[node][i]);
                    visited[In[node][i]]=true;
                }

            if(onlyBack)
                continue;

            for(int i=0; i<Out[node].size(); ++i)
                if(!visited[Out[node][i]])
                {
                    next.push_back(Out[node][i]);
                    visited[Out[node][i]]=true;
                }
         }

         return visited;
     }

     // Checks if the DAG is a single connected component
     bool isConnected()
     {
         vector<int> sources(1, 0);
         vector<bool> reachable = isReachable(sources);

         for(int i=0; i<
         n; ++i)
            if(!reachable[i])
                return false;

         return true;
     }

     // Creates a smaller, 'cleaned' DAG, consisting of only the specified nodes
     DAG keepGivenNodes(const vector<bool>& keepNode)
     {
         int NrOfNodes = 0;
         vector<int> newIdx(n);
         for(int i=0; i<n; ++i)
            if(keepNode[i])
            {
                newIdx[i] = NrOfNodes;
                ++NrOfNodes;
            }

         DAG cleaned(NrOfNodes);
         cleaned.desc=desc;
         for(int i=0; i<n; ++i)
            if(keepNode[i])
                for(int j=0; j<Out[i].size(); ++j)
                    if(keepNode[Out[i][j]])
                        cleaned.addEdge(newIdx[i], newIdx[Out[i][j]], "", true);



         if(DebugMode)
         {
             cout<<"Only the following nodes are kept: ";
             for(int i=0; i<keepNode.size(); ++i)
                if(keepNode[i])
                    cout<<i<<" ";
             cout<<endl;
         }

         return(cleaned);
     }

     void printConnected()
     {
         if(isConnected())
            cout<<"The DAG is connected.\n";
         else
            cout<<"The DAG is NOT connected.\n";
     }
};

struct Matrix
{
    int n;
    int NrNonzeros=0;
    vector< vector<bool> > cells;

    string desc;

    Matrix(int N)
    {
        n=N;
        cells.resize(N, vector<bool>(N,false));
    }

    void Dummy()
    {
        cells.clear();
        cells.resize(n, vector<bool>(n,false));
        for(int i=0; i<n; ++i)
            cells[i][i]=true;

        NrNonzeros = n;

        desc="%%The matrix A is diagonal.";
    }

    // Randomized matrix where each cell is nonzero independently with a fixed probability
    // (if the mainDiag flag is set to true, then the main diagonal is always set to 1)
    void Randomize(double nonzero, bool mainDiag=false)
    {
        for(int i=0; i<n; ++i)
            for(int j=0; j<n; ++j)
            {
                if((i==j && mainDiag) || (double)rand() / (double) RAND_MAX<nonzero)
                {
                    cells[i][j]=true;
                    ++NrNonzeros;

                    if(DebugMode)
                        cout<<"nonzero ("<<i<<","<<j<<")\n";

                }
            }

        char name[200];
        sprintf(name, "%%The probability for having a nonzero in any cell of matrix A is %.2lf", nonzero);
        desc=name;
        if(mainDiag)
            desc+=", and the main diagonal is set to 1.";

        if(DebugMode)
            cout<<"Nonzero count: "<<NrNonzeros<<" out of "<<n*n<<", which is "<<(double)NrNonzeros/((double)n*(double)n)<<" instead of "<<nonzero<<endl;

        // Fallback in case all entries happen to be zero: use a dummy (diagonal) matrix instead
        if(NrNonzeros==0)
        {
            cout<<"The random matrix is completely empty! Using dummy (diagonal) matrix instead.\n";
            Dummy();
        }
    }


    // Reads matrix from file
    bool read(string filename, bool IndexedFromOne = false)
    {
        ifstream infile(filename);
        if(!infile.is_open())
        {
            cout<<"Unable to find/open input matrix file.\n";
            Dummy();
            return false;
        }

        string line;
        getline(infile, line);
        while(!infile.eof() && line.at(0)=='%')
           getline(infile, line);

        int M;
        sscanf(line.c_str(), "%d %d %d", &n, &M, &NrNonzeros);

        if(n != M)
        {
            cout<<"Incorrect input file format: only square matrices are accepted!\n";
            Dummy();
            return false;
        }

        cells.clear();
        cells.resize(n, vector<bool>(n,false));

        // read nonzeros
        int indexOffset = IndexedFromOne ? 1 : 0;
        for(int i=0; i<NrNonzeros; ++i)
        {
            if(infile.eof())
            {
                cout<<"Incorrect input file format (file terminated too early).\n";
                Dummy();
                return false;
            }

            int x, y;
            getline(infile, line);
            while(!infile.eof() && line.at(0)=='%')
               getline(infile, line);

            sscanf(line.c_str(), "%d %d", &x, &y);

            if(x<indexOffset || y<indexOffset || x>=n+indexOffset || y>=n+indexOffset)
            {
                cout<<"Incorrect input file format (index out of range).\n";
                Dummy();
                return false;
            }

            cells[x-indexOffset][y-indexOffset] = true;
        }

        desc="%Matrix A is read from input file "+filename+".";
        infile.close();

        // Check if input matrix is empty, since this can cause undesired behavior in some functions
        if(NrNonzeros==0)
        {
            cout<<"Input matrix is completely empty! Using dummy (diagonal) matrix instead.\n";
            Dummy();
        }

        return true;
     }

    void SetMainDiagonal()
    {
        for(int i=0; i<n; ++i)
            if(!cells[i][i])
            {
                cells[i][i]=true;
                ++NrNonzeros;
            }
    }

    void print( const std::string &label = "", std::ostream& os = std::cout ) const noexcept {
        os << "Matrix " << label
            << ": " << n << "x" << n << ", " << NrNonzeros << " nonzeros\n";
        os << "[\n";
        for(int i=0; i<n; ++i) {
            os << "  ";
            for(int j=0; j<n; ++j) {
                if( cells[i][j] ) {
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
};

DAG CreateRandomIndegExpected(int N, double indeg, double sourceProb = 0.0)
{
    DAG G(N);

    char name[200];
    sprintf(name, "HyperDAG from a random DAG with expected indegree %.2lf.\n%%Each node is chosen as a source with probability %.2lf", indeg, sourceProb);
    G.desc.assign(name);

    for(int i=1; i<N; ++i)
    {
        if( (double)rand() / (double) RAND_MAX < sourceProb )
            continue;

        double p = indeg / (double) i;

        for(int j=0; j<i; ++j)
            if( (double)rand() / (double) RAND_MAX<p)
                G.addEdge(j, i);
    }

    return G;

}


DAG CreateRandomIndegFixed(int N, int indeg, double sourceProb = 0.0)
{
    DAG G(N);

    char name[200];
    sprintf(name, "HyperDAG from a random DAG with fixed indegree %d.\n%%Each node is chosen as a source with probability %.2lf", indeg, sourceProb);
    G.desc.assign(name);

    for(int i=1; i<N; ++i)
    {
        if( (double)rand() / (double) RAND_MAX < sourceProb )
            continue;

        if(i<=indeg) // all previous nodes are chosen
            for(int j=0; j<i; ++j)
                G.addEdge(j, i);

        else // chose 'indeg' predecessors at random
        {
            vector<bool> chosen(i, false);
            for(int j=0; j<indeg; ++j)
            {
                int rnd = randInt(i-j), idx=0;
                for(; chosen[idx]; ++idx) ;
                for(int k=0; k<rnd; ++k)
                {
                    for(++idx; chosen[idx]; ++idx) ;
                }

                G.addEdge(idx, i);
                chosen[idx]=true;
            }
        }

    }

    return G;
}

DAG CreateRandomER(int N, int NrOfEdges)
{
    DAG G(N);

    char name[150];
    sprintf(name, "HyperDAG from a random DAG with expected number of edges %d, with uniform edge probabilities on all forward edges", NrOfEdges);
    G.desc.assign(name);

    double p = 2.0 * (double) NrOfEdges / ((double)N*((double)N-1));

    for(int i=0; i<N; ++i)
        for(int j=i+1; j<N; ++j)
            if( (double)rand() / (double) RAND_MAX<p)
                G.addEdge(i, j);

    return G;
}


DAG CreateSpMV(const Matrix& M)
{
    M.print( "SpMV" );
    int N=M.n;

    DAG G(2*(M.NrNonzeros+N));
    string name="HyperDAG model of SpMV operation.\n"+M.desc;
    G.desc.assign(name);

    // find empty rows in the matrix
    vector<bool> rowNotEmpty(N, false);
    for(int i=0; i<N; ++i)
        for(int j=0; j<N; ++j)
            if(M.cells[i][j])
                rowNotEmpty[i]=true;

    // create SpMV DAG
    int Idx = 0;
    for(int i=0; i<N; ++i) {
        if(!rowNotEmpty[i]) continue;

        for(int j=0; j<N; ++j) {
            if(not M.cells[i][j]) continue;

            // // Operation: u[j] += M[i][j] * v[i]
            // Nodes
            int node_M_i_j = j;
            int node_v_i = N+Idx;
            int node_u_j = N+Idx+1;
            int node_new_u_j = N+2*M.NrNonzeros+i;
            // Labels
            std::string u_new_str = "u'["+to_string(j)+"]";
            std::string u_str = "u["+to_string(j)+"]";
            std::string v_str = "v["+to_string(i)+"]";
            std::string M_str = "M["+to_string(i)+"]["+to_string(j)+"]";
            // Edges
            G.addEdge(node_M_i_j, node_u_j, M_str + " -> " + u_new_str );
            G.addEdge(node_v_i, node_u_j,  u_str + " -> " + u_new_str );
            G.addEdge(node_u_j, node_new_u_j, v_str + " -> " + u_new_str );
            Idx+=2;
        }
    }

    // only keep components that are predecessors of the final nonzeros
    // (in particular, indeces corresponding to (i) empty columns in the original vector or (ii) emtpy rows in the result vector)
    vector<int> sinkNodes;
    for(int i=0; i<N; ++i)
        if(rowNotEmpty[i])
            sinkNodes.push_back(N+2*M.NrNonzeros+i);

    return G.keepGivenNodes(G.isReachable(sinkNodes));
}

DAG CreateRandomSpMV(int N, double nonzero)
{
    Matrix M(N);
    M.Randomize(nonzero);

    DAG G = CreateSpMV(M);

    return G;
}


DAG CreateLLtSolver(const Matrix& M)
{
    M.print( "LLtSolver" );

    DAG G(4*M.n);
    string name="HyperDAG model of LLtSolver operation: inv(trans(L)) . (inv(L) . b)\n"+M.desc;
    G.desc.assign(name);

    // find empty rows in the matrix
    vector<bool> rowNotEmpty(M.n, false);
    for(int i=0; i<M.n; ++i)
        rowNotEmpty[i] = std::any_of(M.cells[i].cbegin(), M.cells[i].cend(), [](const bool x){return x;});
    
    cout << "rowNotEmpty: " << endl;
    for(int i=0; i<M.n; ++i)
        cout << "Row " << i << ": " << rowNotEmpty[i] << endl;

    // Forward substitution DAG
    int LOffset = 0;
    int xOffset = M.n;
    int yOffset = 2*M.n;
    for(int i=0; i<M.n; ++i) {
        if(!rowNotEmpty[i]) continue;

        int y_i = yOffset+i;
        std::string y_i_str = "y["+to_string(i)+"]";

        int x_i = xOffset+i;
        std::string x_i_str = "x["+to_string(i)+"]";
        G.addEdge(x_i, y_i, x_i_str + " -> " + y_i_str );

        // Division by L[i][i]
        int L_i_i = LOffset+i;
        std::string L_i_i_str = "L["+to_string(i)+"]["+to_string(i)+"]";
        G.addEdge(L_i_i, y_i, L_i_i_str + " -> " + y_i_str );

        for(int j=0; j<=i; ++j) {
            if(not M.cells[i][j]) continue;

            // Sum of L[i][k] * x[k]
            for(int k=0; k<j; k++) {
                int L_i_k = LOffset+k;
                std::string L_i_k_str = "L["+to_string(i)+"]["+to_string(k)+"]";
                G.addEdge(L_i_k, y_i, L_i_k_str + " -> " + y_i_str );

                int x_k = xOffset+k;
                std::string x_k_str = "x["+to_string(k)+"]";
                G.addEdge(x_k, y_i, x_k_str + " -> " + y_i_str );
            }
        }
    }
    if(DebugMode) cout << "-- Forward substitution DAG created." << endl;
    
    // Backward substitution DAG
    int zOffset = 3*M.n;
    for(int i=M.n-1; i>=0; --i) {
        if(!rowNotEmpty[i]) continue;

        int y_i = yOffset+i;
        std::string y_i_str = "y["+to_string(i)+"]";

        int z_i = zOffset+i;
        std::string z_i_str = "z["+to_string(i)+"]";
        G.addEdge(y_i, z_i, y_i_str + " -> " + z_i_str );

        // Division by trans(L)[i][i]
        int L_i_i = LOffset+i;
        std::string L_i_i_str = "L["+to_string(i)+"]["+to_string(i)+"]";
        G.addEdge(L_i_i, z_i, L_i_i_str + " -> " + z_i_str );

        for(int j=i+1; j<M.n; ++j) {
            if(not M.cells[i][j]) continue;

            // Sum of L[i][k] * x[k]
            for(int k=1; k<=j; k++) {
                int L_k_i = LOffset+i;
                std::string L_k_i_str = "L["+to_string(k)+"]["+to_string(i)+"]";
                G.addEdge(L_k_i, z_i, L_k_i_str + " -> " + z_i_str );

                int y_k = yOffset+k;
                std::string y_k_str = "y["+to_string(k)+"]";
                G.addEdge(y_k, z_i, y_k_str + " -> " + z_i_str );
            }
        }
        if(DebugMode) cout << endl;
    }
    if(DebugMode) cout << "-- Backward substitution DAG created." << endl;


    // only keep components that are predecessors of the final nonzeros
    // (in particular, indeces corresponding to (i) empty columns in the original vector or (ii) emtpy rows in the result vector)
    vector<int> sinkNodes;
    for(int i=0; i<M.n; ++i)
        if(rowNotEmpty[i])
            sinkNodes.push_back(yOffset + i);

    return G.keepGivenNodes(G.isReachable(sinkNodes));
}

DAG CreateRandomLLtSolver(int N, double nonzero)
{
    Matrix M(N);
    M.Randomize(nonzero);
    DAG G = CreateLLtSolver(M);
    return G;
}

DAG CreateSpMVExp(const Matrix& M, int K)
{
    int N=M.n;

    DAG G((K+1)*(M.NrNonzeros+N));
    string name = "HyperDAG model of naive implementation of A^" + to_string(K) + " *v with sparse matrix A and dense vector v.\n" + M.desc;
    G.desc.assign(name);

    // Index of the nodes representing each nonzero of M
    vector<bool> rowValid(N, true), colValid(N, true);
    vector< vector<int> > originalCellIdx(N, vector<int>(N));
    int Idx=0;
    for(int i=0; i<N; ++i)
        for(int j=0; j<N; ++j)
            if(M.cells[i][j])
            {
                originalCellIdx[i][j]=Idx;
                ++Idx;
            }

    // Vbase is the starting point of the current vector v, base is the start index for the next iteration
    int Vbase = M.NrNonzeros, base = Vbase+N;


    // ITERATIONS
    for(int k=0; k<K; ++k)
    {
        colValid = rowValid;

        // count the number of used cells in current iteration
        int usedCells=0;
        for(int i=0; i<N; ++i)
            for(int j=0; j<N; ++j)
                if(M.cells[i][j] && colValid[j])
                    ++usedCells;

        if(usedCells==0)
        {
            cout<<"Error: the whole matrix has vanished (we get a zero vector after "<<k+1<<" iterations)!\n";
            cout<<"The output hyperDAG corresponds to the result after "<<k<<" iterations.\n";
            break;
        }

        // iterate over cells, add DAG edges
        Idx=0;
        rowValid.assign(N, false);
        for(int i=0; i<N; ++i)
            for(int j=0; j<N; ++j)
                if(M.cells[i][j] && colValid[j])
                {
                    G.addEdge(Vbase+j, base+Idx);
                    G.addEdge(originalCellIdx[i][j], base+Idx);
                    G.addEdge(base+Idx, base+usedCells+i);
                    ++Idx;
                    rowValid[i] = true;
                }

        Vbase = base + usedCells;
        base = Vbase + N;
    }

    // only keep components that are predecessors of the final nonzeros
    // (this can remove many intermediate values that were never used: they were computed, but then they were only multiplied by zeros)
    vector<int> finalVector;
    for(int i=0; i<N; ++i)
        if(rowValid[i])
            finalVector.push_back(Vbase+i);

    return G.keepGivenNodes(G.isReachable(finalVector, true));
}

DAG CreateRandomSpMVExp(int N, double nonzero, int K)
{
    Matrix M(N);
    M.Randomize(nonzero);

    DAG G = CreateSpMVExp(M, K);

    return G;
}

DAG CreateSparseCG(const Matrix& M, int K)
{
    int N=M.n;

    DAG G(3*N-1+2*M.NrNonzeros+K*(M.NrNonzeros+9*N+4));
    string name = "HyperDAG model of naive implementation of " + to_string(K) + " iterations of the conjugate gradient method.\n" + M.desc;
    G.desc.assign(name);

    // Index of the nodes representing each nonzero of M
    // (we will denote the matrix M by "A" in the comment pseudocodes, since it is more standard)
    vector< vector<int> > originalCellIdx(N, vector<int>(N));
    int Idx=0;
    for(int i=0; i<N; ++i)
        for(int j=0; j<N; ++j)
            if(M.cells[i][j])
            {
                originalCellIdx[i][j]=Idx;
                ++Idx;
            }

    // INITIALIZATION PHASE

    // compute A * x_0
    Idx = 0;
    for(int i=0; i<N; ++i)
        for(int j=0; j<N; ++j)
        {
            if(M.cells[i][j])
            {
                G.addEdge(M.NrNonzeros+j, M.NrNonzeros+N+Idx);
                G.addEdge(originalCellIdx[i][j], M.NrNonzeros+N+Idx);
                G.addEdge(M.NrNonzeros+N+Idx, 2*M.NrNonzeros+N+i);
                Idx+=1;
            }
        }

    // Rbase, Pbase and Xbase points to the beginning of the current r, p and x vectors
    int Rbase = 2*M.NrNonzeros+3*N, Pbase = Rbase+N, Xbase = M.NrNonzeros;

    // r_0 := b - A * x_0 ;  p_0 := r_0
    for(int i=0; i<N; ++i)
    {
        G.addEdge(2*M.NrNonzeros+N+i, Rbase+i);
        G.addEdge(2*M.NrNonzeros+2*N+i, Rbase+i);
        G.addEdge(Rbase+i, Pbase+i);
    }

    // compute r_0^T * r_0 and save its index in Rproduct
    int Rproduct = Pbase+2*N;
    for(int i=0; i<N; ++i)
    {
        G.addEdge(Rbase+i, Pbase+N+i);
        G.addEdge(Pbase+N+i, Rproduct);
    }

    // ITERATIONS

    // base denotes the current index at the beginning of each iteration
    int base=Rproduct+1;

    for(int k=0; k<K; ++k)
    {
        // compute A * p_k
        Idx = 0;
        for(int i=0; i<N; ++i)
            for(int j=0; j<N; ++j)
            {
                if(M.cells[i][j])
                {
                    G.addEdge(Pbase+j, base+Idx);
                    G.addEdge(originalCellIdx[i][j], base+Idx);
                    G.addEdge(base+Idx, base+M.NrNonzeros+i);
                    Idx+=1;
                }
            }

        // compute p_k^T * A * p_k
        for(int i=0; i<N; ++i)
        {
            G.addEdge(Pbase+i, base+M.NrNonzeros+N+i);
            G.addEdge(base+M.NrNonzeros+i, base+M.NrNonzeros+N+i);
            G.addEdge(base+M.NrNonzeros+N+i, base+M.NrNonzeros+2*N);
        }

        // alpha_k := r_k^T * r_k / p_k^T * A * p_k
        G.addEdge(base+M.NrNonzeros+2*N, base+M.NrNonzeros+2*N+1);
        G.addEdge(Rproduct, base+M.NrNonzeros+2*N+1);

        // index of alpha_k
        int alpha = base+M.NrNonzeros+2*N+1;

        // x_(k+1) := x_k + alpha_k * p_k
        for(int i=0; i<N; ++i)
        {
            G.addEdge(Pbase+i, alpha+1+i);
            G.addEdge(alpha, alpha+1+i);
            G.addEdge(alpha+1+i, alpha+1+N+i);
            G.addEdge(Xbase+i, alpha+1+N+i);
        }
        Xbase = alpha+1+N;

        // r_(k+1) := r_k - alpha_k * A * p_k
        for(int i=0; i<N; ++i)
        {
            G.addEdge(base+M.NrNonzeros+i, Xbase+N+i);
            G.addEdge(alpha, Xbase+N+i);
            G.addEdge(Xbase+N+i, Xbase+2*N+i);
            G.addEdge(Rbase+i, Xbase+2*N+i);
        }
        Rbase = Xbase+2*N;

        if(k==K-1)
            break;

        // compute r_(k+1)^T * r_(k+1)
        for(int i=0; i<N; ++i)
        {
            G.addEdge(Rbase+i, Rbase+N+i);
            G.addEdge(Rbase+N+i, Rbase+2*N);
        }

        // beta_k := r_(k+1)^T * r_(k+1) / r_k^T * r_k
        G.addEdge(Rbase+2*N, Rbase+2*N+1);
        G.addEdge(Rproduct, Rbase+2*N+1);

        // updated index of Rproduct and beta_k
        Rproduct = Rbase+2*N;
        int beta = Rbase+2*N+1;

        // p_(k+1) := r_(k+1) - beta_k * p_k
        for(int i=0; i<N; ++i)
        {
            G.addEdge(Pbase+i, beta+1+i);
            G.addEdge(beta, beta+1+i);
            G.addEdge(beta+1+i, beta+1+N+i);
            G.addEdge(Rbase+i, beta+1+N+i);
        }

        Pbase = beta+1+N;
        base = Pbase+N;

    }

    return G;

}

DAG CreateRandomSparseCG(int N, double nonzero, int K)
{
    Matrix M(N);
    M.Randomize(nonzero);

    DAG G = CreateSparseCG(M, K);

    return G;
}


DAG CreatekNN(const Matrix& M, int K, int source)
{
    int N=M.n;

    DAG G((K+1)*(M.NrNonzeros+N));
    string name = "HyperDAG model of naive implementation of " + to_string(K) + " iterations of kNN, starting from node number " + to_string(source) + " (i.e. sparse vector with 1 entry).\n" + M.desc;
    G.desc.assign(name);

    // Index of the nodes representing each nonzero of M
    vector< vector<int> > originalCellIdx(N, vector<int>(N));
    int Idx=0;
    for(int i=0; i<N; ++i)
        for(int j=0; j<N; ++j)
            if(M.cells[i][j])
            {
                originalCellIdx[i][j]=Idx;
                ++Idx;
            }

    // Initialize sparse vector
    vector<bool> reached(N, false);
    reached[source]=true;

    // Vbase is the current index of the beginning of the vector
    int Vbase = Idx;
    // base is the index where the next iteration starts
    int base = Idx + 1;

    // current new index for each original row of the sparse vector
    vector<int> newRowIdx(N);
    newRowIdx[source] = 0;

    // Iterations
    for(int k=0; k<K; ++k)
    {
        // count the number of internal nodes (matrix-vector multiplication) in this iteration
        int NrOfInternalNodes = 0;
        for(int i=0; i<N; ++i)
            for(int j=0; j<N; ++j)
                if(reached[j] && M.cells[i][j])
                    ++NrOfInternalNodes;

        // add DAG edges
        int rowIdx = 0, cellIdx = 0;
        vector<bool> rowNotEmpty(N, false);
        for(int i=0; i<N; ++i)
        {
            for(int j=0; j<N; ++j)
                if(reached[j] && M.cells[i][j])
                {
                    G.addEdge(Vbase+newRowIdx[j], base+cellIdx);
                    G.addEdge(originalCellIdx[i][j], base+cellIdx);
                    G.addEdge(base+cellIdx, base+NrOfInternalNodes+rowIdx);
                    cellIdx+=1;
                    rowNotEmpty[i]=true;
                }

            if(rowNotEmpty[i])
                ++rowIdx;
        }

        // setup for next iteration
        rowIdx = 0;
        for(int i=0; i<N; ++i)
            if(rowNotEmpty[i])
            {
                reached[i]=true;
                newRowIdx[i]=rowIdx;
                ++rowIdx;
            }

        int oldbase = base;
        base = oldbase+NrOfInternalNodes+rowIdx;
        Vbase = oldbase+NrOfInternalNodes;
    }

    // only keep components that are connected to the final nonzeros
    // (in particular, this removes nonzeros of the matrix that were never used)
    vector<int> finalVector;
    for(int i=Vbase; i<base; ++i)
        finalVector.push_back(i);

    return G.keepGivenNodes(G.isReachable(finalVector));
}


DAG CreateRandomkNN(int N, double nonzero, int K)
{
    Matrix M(N);
    M.Randomize(nonzero, true);

    int source = randInt(N);

    DAG G = CreatekNN(M, K, source);

    return G;
}


int main(int argc, char* argv[])
{
    int N=-1, edges=-1, K=-1, sourceNode=-1, indegFix=-1;
    double sourceProb=-1.0, nonzeroProb=-1.0, indegExp = -1.0;
    string infile, outfile, mode, indegreeString;
    bool indexedFromOne = false;

    // PROCESS COMMAND LINE ARGUMENTS

    vector<string> params{"-output", "-input", "-mode", "-N", "-K", "-edges", "-indegree", "-sourceProb", "-nonzeroProb", "-sourceNode"};
    vector<string> modes{"ER", "fixedIn", "expectedIn", "SpMV", "SpMVExp", "LLtSolver", "kNN", "CG"};

    for (int i = 1; i < argc; ++i)
    {
        // Check parameters that require an argument afterwards
        if (contains(params, string(argv[i])) && i + 1 >= argc)
        {
            cerr << "Parameter error: no parameter value after the \""<<string(argv[i])<<"\" option." << endl;
            return 1;
        }

        if (string(argv[i]) == "-output")
            outfile = argv[++i];

        else if (string(argv[i]) == "-input")
            infile = argv[++i];

        else if (string(argv[i]) == "-mode")
        {
            mode = argv[++i];
            if(!contains(modes, mode))
            {
                cerr << "Parameter error: invalid mode parameter. Please check the readme for possible modes." << endl;
                return 1;
            }
        }

        else if (string(argv[i]) == "-N")
        {
            N = stoi(argv[++i]);
            if(N<=1)
            {
                cerr << "Parameter error: N has to be at least 2." << endl;
                return 1;
            }
        }

        else if (string(argv[i]) == "-K")
        {
            K = stoi(argv[++i]);
            if(K<=0)
            {
                cerr << "Parameter error: K has to be at least 1." << endl;
                return 1;
            }
        }

        else if (string(argv[i]) == "-edges")
            edges = stoi(argv[++i]);

        else if (string(argv[i]) == "-indegree")
            indegreeString = argv[++i];

        else if (string(argv[i]) == "-sourceProb")
        {
            sourceProb = stod(argv[++i]);
            if(sourceProb<-0.0001 || sourceProb>1.0001)
            {
                cerr << "Parameter error: parameter \"sourceProb\" has to be between 0 and 1." << endl;
                return 1;
            }
        }

        else if (string(argv[i]) == "-nonzeroProb")
        {
            nonzeroProb = stod(argv[++i]);
            if(nonzeroProb<-0.0001 || nonzeroProb>1.0001)
            {
                cerr << "Parameter error: parameter \"nonzeroProb\" has to be between 0 and 1." << endl;
                return 1;
            }
        }

        else if (string(argv[i]) == "-sourceNode")
        {
            sourceNode = stoi(argv[++i]);
            if(sourceNode<0)
            {
                cerr << "Parameter error: sourceNode cannot be negative." << endl;
                return 1;
            }
        }

        else if (string(argv[i]) == "-debugMode")
            DebugMode = true;

        else if (string(argv[i]) == "-indexedFromOne")
            indexedFromOne = true;

        else
        {
            cerr << "Parameter error: unknown parameter/option "<< string(argv[i]) << endl;
            return 1;
        }
    }


    // CHECK PARAMETER CONSISTENCY

    // mode, infile, outfile
    if(mode.empty())
    {
        cerr << "Parameter error: no mode specified." << endl;
        return 1;
    }
    if(!infile.empty() && (mode=="ER" || mode=="fixedIn" || mode== "expectedIn"))
    {
        cerr << "Parameter error: cannot use input matrix file for this mode." << endl;
        return 1;
    }
    if(outfile.empty())
    {
        outfile="output.txt";
        if(DebugMode)
            cout << "Output file not specified; using default output filename (output.txt)." << endl;
    }
    if(infile.empty() && indexedFromOne)
    {
        cerr << "Parameter error: cannot use parameter \"indexedFromOne\" without an input file." << endl;
        return 1;
    }

    // N
    if(!infile.empty())
    {
        if(N!=-1)
        {
            cerr << "Parameter error: cannot use parameter \"N\" if input file is specified." << endl;
            return 1;
        }
    }
    // If not set, use 10 as default value
    else if(N==-1)
        N = 10;


    // K
    if(mode=="ER" || mode=="fixedIn" || mode=="expectedIn" || mode=="SpMV" || mode=="LLtSolver" )
    {
        if(K>=0)
        {
            cerr << "Parameter error: cannot use parameter \"K\" for this mode." << endl;
            return 1;
        }
    }
    else
    {
        // If not set, use 3 as default value
        if(K<0)
            K = 3;
    }

    // edges
    if(mode!="ER")
    {
        if(edges>=0)
        {
            cerr << "Parameter error: cannot use parameter \"edges\" for this mode." << endl;
            return 1;
        }
    }
    // If not set, use 5*N as default value
    else if(edges<0)
        edges = N*(N-1)/5;



    //indegree
    if(mode!="fixedIn" && mode!="expectedIn")
    {
        if(!indegreeString.empty())
        {
            cerr << "Parameter error: cannot use parameter \"indegree\" for this mode." << endl;
            return 1;
        }
    }
    else if(mode=="fixedIn")
    {
        // If not set, use 4 as default value
        if(indegreeString.empty())
            indegFix=4;
        else
            indegFix=stoi(indegreeString);

        if(indegFix<=0 || indegFix>=N)
        {
            cerr << "Parameter error: parameter \"indegree\" has to be between 1 and (N-1)." << endl;
            return 1;
        }
    }
    else if(mode=="expectedIn")
    {
        // If not set, use 4 as default value
        if(indegreeString.empty())
            indegExp=4.0;
        else
            indegExp=stod(indegreeString);

        if(indegExp<=-0.0001 || indegExp>=N-0.0009)
        {
            cerr << "Parameter error: parameter \"indegree\" has to be between 0 and (N-1)." << endl;
            return 1;
        }
    }


    // sourceProb
    if(mode!="fixedIn" && mode!="expectedIn")
    {
        if(sourceProb>-0.5)
        {
            cerr << "Parameter error: cannot use \"sourceProb\" parameter for this mode." << endl;
            return 1;
        }
    }
    // If not set, use 0 as default value
    else if(sourceProb<-0.5)
        sourceProb = 0.0;


    // nonzeroProb
    if(mode=="ER" || mode=="fixedIn" || mode=="expectedIn")
    {
        if(nonzeroProb>-0.5)
        {
            cerr << "Parameter error: cannot use parameter \"nonzeroProb\" for this mode." << endl;
            return 1;
        }
    }
    else
    {
        if(nonzeroProb>-0.5 && !infile.empty())
        {
            cerr << "Parameter error: parameter \"nonzeroProb\" has no use if input file is specified." << endl;
            return 1;
        }
        // If not set, use 0.25 as default value
        if(nonzeroProb<-0.5)
            nonzeroProb = 0.25;
    }

    // sourceNode
    if(sourceNode>=0 && mode!="kNN")
    {
        cerr << "Parameter error: \"sourceNode\" parameter can only be used in kNN mode." << endl;
        return 1;
    }
    else if(sourceNode>=0 && infile.empty())
    {
        cerr << "Parameter error: \"sourceNode\" parameter is only used when the matrix is read from a file." << endl;
        return 1;
    }
    // Use 0 as default value
    else if(mode=="kNN" && sourceNode==-1)
        sourceNode = 0;


    // INITIALIZE
    srand(0);
    Matrix M(1);
    if(!infile.empty())
    {
        if(!M.read(infile, indexedFromOne))
        {
            cerr << "Error reading matrix from input file." << endl;
            return 1;
        }
        N = M.n;
    }

    // last parameter check (has to be done after N is read from file)
    if(mode=="kNN" && sourceNode>=N)
    {
        cout << "Parameter error: sourceNode is a node index, it cannot be larger than (N-1)." << endl;
        return 1;
    }


    // GENERATE HYPERDAG

    DAG G(1);

    if(mode=="ER")
    {
        G = CreateRandomER(N, edges);
    }
    else if(mode=="fixedIn")
    {
        G = CreateRandomIndegFixed(N, indegFix, sourceProb);
    }
    else if(mode=="expectedIn")
    {
        G = CreateRandomIndegExpected(N, indegExp, sourceProb);
    }
    else if(mode=="SpMV")
    {
        if(!infile.empty())
            G = CreateSpMV(M);
        else
            G = CreateRandomSpMV(N, nonzeroProb);
    }
    else if(mode=="SpMVExp")
    {
        if(!infile.empty())
            G = CreateSpMVExp(M, K);
        else
            G = CreateRandomSpMVExp(N, nonzeroProb, K);
    }
    else if(mode=="LLtSolver")
    {
        if(!infile.empty())
            G = CreateLLtSolver(M);
        else
            G = CreateRandomLLtSolver(N, nonzeroProb);
    }
    else if(mode=="kNN")
    {
        if(!infile.empty())
        {
            M.SetMainDiagonal();
            G = CreatekNN(M, K, sourceNode);
        }
        else
            G = CreateRandomkNN(N, nonzeroProb, K);
    }
    else if(mode=="CG")
    {
        if(!infile.empty())
            G = CreateSparseCG(M, K);
        else
            G = CreateRandomSparseCG(N, nonzeroProb, K);
    }

    if(DebugMode)
        G.printConnected();

    G.printHyperDAG(outfile);
    return 0;
}
