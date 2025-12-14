---------
COPYRIGHT
---------

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

---------------------
HYPERDAG GENERATOR V1
---------------------

This is the HyperDAG generator tool (version 1) created at Computational Systems Laboratory, Zurich Research Center, Huawei Technologies Switzerland.

Disclaimer: the tool is a research prototype, and as such, it might occasionally contain unexpected bugs or runtime errors. It also assumes that it is run with a correct parametrization: the correctness of the program parameters, and the correct format of the input file are only checked to a limited extent.

--------
PURPOSE:
--------

The main purpose of the tool is to generate the hyperDAG representation of several forms of computational DAGs. The tool has seven different modes to generate a specific kind of computational DAG (and then the corresponding hyperDAG from this).

Three of these modes correspond to random DAG models, where the nodes are ordered from 0 to (N-1), and edges are added from node i to node j (with i<j) according to different random rules: the ER mode resembles classical random graph formats, while the fixedIn and expectedIn modes describe a different random model that is inspired by some practical properties of real computational DAGs.

The other four modes correspond to fine-grained representations of specific matrix-based computations: (iterated) matrix-vector multiplications or the conjugate gradient method. In these cases, all nonzeros of the initial matrices/vectors are represented by a separate node of the DAG, and the further nodes of the DAG correspond to further intermediate/output values during the given computation. We note that the computational DAGs in these modes describe a rather basic implementation of the given algorithms, without e.g. any low-level optimizations or such.

-------
OUTPUT:
-------

A computational hyperDAG written into a file. The format of the output file is as follows.
- The first few lines, starting with a '%' character, are comment lines giving a brief description of the hyperDAG (and if applicable, the corresponding matrix).
- The next line contains three integers "M N P": the number of hyperedges M, the number of nodes N, and the number of pins P in the hyperDAG in this order.
- The following M lines each contain two integers "H W", where H is a hyperedge index between 0 and M-1, and W is the weight associated with this hyperedge (its "communication cost"). In our generator, these weights are uniformly 1.
- The following N lines each contain two integers "V W", where V is a node index between 0 and N-1, and W is the weight associated with this node (its "work cost"). In our generator, the work cost of each node is obtained as its indegree minus 1 (but at least 0).
- The following P lines each describe a specific pin, containing two numbers: the first is the index of the hyperedge, the second is the index of the node.

------
MODES:
------

The generator supports the following modes of generating a random hyperDAG:

ER: generates an Erdos-Renyi style random DAG. For each 0<=i<j<=(N-1), there is an edge from node i to node j with a probability p independently. p is chosen such that the expected number of edges in the DAG is equal to the "edges" parameter. Possible parameters: -output, -N, -edges

fixedIn: generates a random DAG where each node has an indegree that is equal to the "indegree" parameter. For node i, the "indegree" edges coming to node i are chosen uniformly from nodes 0,1,...,(i-1). If i<"indegree", the node i has an edge from all previous nodes, and its degree is only i. Additionally, if the sourceProb parameter is used, then each node is chosen as a source with this given probability, meaning that it does not have any incoming edges at all (corresponding to e.g. new input scalars in a real-world computation). Possible parameters: -output, -N, -indegree, -sourceProb

expectedIn: generates a random DAG where each node has an indegree that is equal to the "indegree" parameter in expectation. That is, for node i, there is an edge from each node 0,1,...,(i-1) to node i independently with a fixed probability p, and p is always chosen such that the expected number of incoming edges of node i is "indegree". If i<"indegree", the node i has an edge from all previous nodes, and its degree is only i. Additionally, if the sourceProb parameter is used, then each node is chosen as a source with this given probability, meaning that it does not have any incoming edges at all (corresponding to e.g. new input scalars in a real-world computation). Possible parameters: -output, -N, -indegree, -sourceProb

SpMV: generates a DAG that corresponds to the fine-grained representation of a sparse matrix - dense vector multiplication (out-of-place). By default, it generates an input matrix A of size NxN, where each cell is independently nonzero with a given probability, specified by the "nonzeroProb" parameter. The matrix A is then multiplied by a dense vector of length N. In the end, nodes of the DAG which are no connected to the final vector (e.g. cells of the vector that correspond to a matrix column that is entirely zero) are removed. It is also possible to read the input matrix from a file if the "input" parameter is used. Possible parameters: -output, -N, -input, -nonzeroProb

SpMVExp: generates a DAG that corresponds to the fine-grained representation of the operation where we multiply a sparse matrix A by a dense vector v, K times consecutively (A^k * v). By default, it generates an input matrix A of size NxN, where each cell is independently nonzero with a given probability, specified by the "nonzeroProb" parameter. The matrix A is then multiplied by a dense vector (of length N) K times consecutively. In the end, nodes of the DAG which are not predecessors of a nonzero value in the final vector are removed. Note that this step can remove larger parts of the DAG when some intermediate values are computed, but then always only multiplied by zeros (in some cases, the number of nodes in the DAG can even decrease when K is incremented!). It is also possible to read the input matrix from a file if the "input" parameter is used. Possible parameters: -output, -N, -K, -input, -nonzeroProb

LLtSolver: generates a DAG that corresponds to the fine-grained representation of: z =  inv(trans(L)) . (inv(L) . x). By default, it generates an input matrix A of size NxN, where each cell is independently nonzero with a given probability, specified by the "nonzeroProb" parameter. The matrix L (lower triangular matrix) is then multiplied by a dense vector of length N, then the result is re-multiplied by this same matrix L but transposed. In the end, nodes of the DAG which are no connected to the final vector (e.g. cells of the vector that correspond to a matrix column that is entirely zero) are removed. It is also possible to read the input matrix from a file if the "input" parameter is used. Possible parameters: -output, -N, -input, -nonzeroProb

LUSolver: generates a DAG that corresponds to the fine-grained representation of: z =  inv(U) . (inv(L) . x). By default, it generates an input matrix A of size NxN, where each cell is independently nonzero with a given probability, specified by the "nonzeroProb" parameter. The matrix L (lower triangular matrix) is then multiplied by a dense vector of length N, then the result is re-multiplied by this same matrix L but transposed. In the end, nodes of the DAG which are no connected to the final vector (e.g. cells of the vector that correspond to a matrix column that is entirely zero) are removed. It is also possible to read the input matrix from a file if the "input" parameter is used. Possible parameters: -output, -N, -input, -nonzeroProb

kNN: generates a DAG that corresponds to the fine-grained representation of the operation where we multiply a sparse matrix A by a sparse vector v, K times consecutively (A^k * v). By default, it generates an input matrix A of size NxN, where each cell is independently nonzero with a given probability, specified by the "nonzeroProb" parameter. The vector "v" only has one nonzero cell (at a randomly chosen index) in the beginning. The matrix A is then multiplied by the vector v, and this is done K times consecutively. In the end, nodes of the DAG which are no connected to the final vector are removed. It is also possible to read the input matrix from a file if the "input" parameter is used; in this case, we can also specify the index of v where the nonzero cell is located, through the "sourceNode" parameter. Possible parameters: -output, -N, -K, -input, -nonzeroProb, -sourceNode

CG: generates a DAG that corresponds to the fine-grained representation of K iterations of the conjugate gradient method for a sparse matrix A (corresponding to a simple version of the iterative method, as described in the pseudocode in the appropriate Wikipedia article). By default, it generates an input matrix A of size NxN, where each cell is independently nonzero with a given probability, specified by the "nonzeroProb" parameter. It is also possible to read the input matrix from a file if the "input" parameter is used. Possible parameters: -output, -N, -K, -input, -nonzeroProb

-------------------------------------
POSSIBLE PARAMETERS (case sensitive):
-------------------------------------

"-mode" : compulsory parameter, describes the method used for generating a hyperDAG. Possible values are: ER, fixedIn, expectedIn, SpMV, SpMVExp, LLtSolver, kNN, CG

"-output": describes the name of the output file where the hyperDAG is written. If not specified, the default name is "output.txt"

"-input": if used, then instead of randomly generating a matrix, the input matrix is read from a text file. The input parameter describes the name of the input file to read the matrix from. The input file is expected to be in a MatrixMarket-like "coordinate pattern general" format (briefly described at the end of this file), with columns/rows indexed from 0 by default. Can only be used in fine-grained (SpMV, SpMVExp, kNN, CG) modes.

"-N" : describes the number of rows/columns in the square matrix (in SpMV, SpMVExp, kNN and CG mode) or the number of nodes in the random DAG (ER, fixedIN and expectedIn mode). If not specified, the default value is 10 in any mode.

"-K" : describes the number of iterations executed by a specific algorithm for the appropriate fine-grained modes. If not specified, the default value is 3. Can only be used in modes SpMVExp, kNN and CG.

"-nonzeroProb" : describes the probability of each cell of the matrix being a nonzero for the fine-grained modes. Its value has to be between 0 and 1. If not specified, the default value is 0.25. Can only be used in fine-grained (SpMV, SpMVExp, kNN, CG) modes.

"-edges" : describes the number of expected edges in the DAG in ER mode; the edge probability p is calculated accordingly. If not specified, the default value is N*(N-1)/5. Can only be used in ER mode.

"-indegree" : describes the number of incoming edges for each node in fixedIn mode, and the expected number of incoming edges for each node in expectedIn mode. Has to be between 1 and (N-1) in fixedIn mode, and between 0 and (N-1) in expectedIn mode. If not specified, the default value is 4. Can only be used in fixedIn and expectedIn modes.

"-sourceProb" : describes the probability for each node to be chosen as a source with indegree 0. Its value has to be between 0 and 1. If not specified, the default value is 0. Can only be used in fixedIn and expectedIn modes.

"-sourceNode" : in kNN mode, it specifies the cell index of the sparse vector that is initially non-zero. Has to be an integer between 0 and (N-1). Can only be used in kNN mode if an input file is specified.

"indexedFromOne" : if set, then the input matrix file is assumed to have the matrix rows/columns indexed from 1 to N. By default (if this flag is not used), we assume that the rows/columns of the input file are indexed from 0 to (N-1). See a more detailed overview of the input file format below. This parameter does not need a following argument. It can only be used in fine-grained (SpMV, SpMVExp, kNN, CG) modes, and only if an input file is specified.

"-debugMode": enables debugging messages on the standard output during the run of the program. This parameter does not need a following argument, and it can be used in any mode.

-------------------
SOME EXAMPLE CALLS:
-------------------
  
-mode ER  
-mode ER -N 50 -edges 200 -output outfile.txt  
-mode fixedIn -indegree 3 -N 20  
-mode fixedIn -N 40 -indegree 6 -sourceProb 0.1 -debugMode  
-mode expectedIn -N 17 -indegree 2.5 -sourceProb 0.2  
-mode SpMV -input infile.txt -output outfile.txt -debugMode  
-mode SpMV -N 10 -nonzeroProb 0.33  
-mode SpMVExp -K 10 -nonzeroProb 0.2  
-mode SpMVExp -K 5 -input infile.txt  
-mode kNN -N 15 -K 3 -nonzeroProb 0.25  
-mode kNN -input infile.txt -indexedFromOne -sourceNode 3 
-mode CG -N 20 -K 5 -nonzeroProb 0.4 -debugMode  
-mode CG -input infile.txt -K 1  

---------------------------------
INPUT FILE FORMAT (for matrices):
---------------------------------

If the "input" option is used to read the matrix from an input file, then the input file is expected in a MatrixMarket-like "coordinate pattern general" format, as briefly summarized below.

- The file can begin with several header/comment lines, each starting with a '%'.

- The first non-comment line should contain three integers "N N Z" separated by spaces, where N is the number of columns/rows, and the third number is the number of nonzeros. Note that the first two numbers need to be equal; we are assuming a square matrix, but N still has to appear twice in this line to conform to standards.

- The next Z lines should each contain two integers "X Y", separated by a space, meaning that there is a non-zero in row X, column Y. Additionally, the value of the nonzero can appear at the end of the line, separated by another whitespace from Y; this will be ignored.

- By default, we assume that rows and columns are numbered from 0 to (N-1), to conform to general notations on graphs and to our output files. However, several matrix collections (e.g. MatrixMarket) instead have rows and columns numbered from 1 to N. In order to read matrices with such indexing, the "indexedFromOne" flag can be used.
