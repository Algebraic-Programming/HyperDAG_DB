
The hyperDAG backend is implemented in ALP/GraphBLAS, freely available from v0.7
onwards. The below in part derives from its README description-- for its most up
to date version, please see the ALP/GraphBLAS repository on GitHub or Gitee. All
hyperDAGs in this database are created using this hyperDAG backend and by using
the standard algorithm drivers bundled with ALP/GraphBLAS. A summary of the
algorithms this database contains follows [in the last section](#contents).

The ALP/GraphBLAS hyperDAGs backend
===================================

An ALP/GraphBLAS backend may be viewed as an implementation of the C++11
ALP/GraphBLAS API. Backends may target different architectures and systems, such
as a sequential machine, a shared-memory parallel one, a hybrid shared- and
distributed-memory parallel one, or one for a specific accelerator. Multiple
backends may be compiled into a single binary. Thus, ALP/GraphBLAS executables
may be suitable for heterogeneous systems, while ALP/GraphBLAS executables may
be compositional. This latter property is key to the `hyperdags` backend here
described.

The `hyperdags` backend gathers meta-data while user programs execute. The actual
compute logic is executed by a compile-time selected secondary backend, which by
default is the sequential auto-vectorising `reference` backend. The meta-data
will be used to generate, at program exit, a hyperDAG representation of the
executed computation. We foresee two possible hyperDAG representations:

 1. a coarse-grain representation where vertices correspond to a) source
    containers, b) output containers, or c) ALP/GraphBLAS primitives (such 
    as grb::mxv or grb::dot). Hyperedges capture which vertices act as source
    to operations or outputs in other vertices. Each hyperedge has exactly
    one source vertex only.

 2. a fine-grain representation where source vertices correspond to nonzeroes
    in a source container, not the container as a whole, and likewise for output
    vertices that correspond to individual elements of output containers. Also
    there are now many fine-grained operation vertices that are executed by a
    single ALP/GraphBLAS primitive. For example, a call to grb::vxm will emit
    two hyperedges for every nonzero in the sparse input matrix.

Only the extraction of a coarse-grained representation is presently implemented.

Usage
-----

To use the hyperDAG generation backend, follow the following steps. Note that
steps 1-5 are common to building the general ALP/GraphBLAS template library.
Steps 6 & 7 showcase the HyperDAG generation using representation no. 1 on the
tests/smoke/kmeans.cpp. This algorithm has an embedded datase, so there is no
need of giving an input.

1. `cd /path/to/ALP/GraphBLAS/root/directory`

2. `./configure --prefix=/path/to/install/directory`

3. `cd build`

4. `make -j && make -j install`

5. `source /path/to/install/directory/bin/setenv`

6. `grbcxx -b hyperdags -g -O0 -Wall -o dot_hyperdag ../tests/smoke/kmeans.cpp`

7. `grbrun -b hyperdags ./kmeans_hyperdag`

Explanation
-----------

After the above steps, something like the following will be produced:

```
Functional test executable: ./kmeans_hyperdags
Info: grb::init (hyperdags) called.
Info: grb::init (reference) called.
	kmeans converged successfully after 1 iterations.
Info: grb::finalize (hyperdags) called.
	 dumping HyperDAG to stdout
%%MatrixMarket weighted-matrix coordinate pattern general
%	 There are 4 unique source vertex types present in this graph. An index of source type ID and their description follows:
%		 0: input scalar
%		 1: ALP/GraphBLAS container
%		 2: input iterator
%		 3: input integer
%	 There are 13 unique operation vertex types present in this graph. An index of vertex type ID and their description follows:
%		 6: clear( vector )
%		 7: setElement( vector )
%		 9: set( vector, scalar )
%		 12: set( vector, vector )
%		 50: mxm( matrix, matrix, matrix, semiring, scalar )
%		 51: mxm( matrix, matrix, matrix, monoid, scalar, scalar )
%		 52: outer( matrix, vector, vector, scalar, scalar )
%		 65: vxm( vector, vector, matrix, ring )
%		 69: mxv( vector, matrix, vector, ring )
%		 71: buildMatrixUnique( matrix, scalar, scalar, scalar )
%		 75: resize( matrix, scalar )
%		 81: vxm( vector, vector, matrix, scalar, scalar )
%		 86: foldl( vector, vector, monoid)
%	 There are 1 unique output vertex types present in this graph. An index of output vertex type ID and their description follows:
%		 103: output container
80 87 94

```

This output contains the HyperDAG corresponding to the code in the given source file,
tests/unit/dot.cpp. Let us examine it. First, ALP/GraphBLAS will always print info 
(and warning) statements to the standard error stream. These are:

```
Functional test executable: ./kmeans_hyperdags
Info: grb::init (hyperdags) called.
Info: grb::init (reference) called.
	kmeans converged successfully after 1 iterations.
Info: grb::finalize (hyperdags) called.
	 dumping HyperDAG to stdout
```

These statements indicate which backends are used and when they are initialised,
respectively, finalised. The info messages indicate that the hyperdags backend is 
used, which, in turn, employs the standard sequential reference backend for the actual
computations. The second to last message reports that as part of finalising 
the hyperdags backend, it dumps the HyperDAG constructed during computations
to the stdandard output stream (stdout).

The output to stdout starts with:

```
%%MatrixMarket matrix coordinate pattern general

```

This indicates the HyperDAG is stored using a MatrixMarket
(https://dl.acm.org/doi/10.1145/2049662.2049663) format. As the name implies,
this format stores sparse matrices, so we need a definition of how the sparse matrix is 
mapped back to a HyperDAG. Here, rows correspond to hyperedges while columns correspond
to vertices. In the MatrixMarket format, comments are allowed and should start with a %.
The hyperdags backend presently prints which vertices are sources as comment lines.

Then the output continues with:

```
There are 4 unique source vertex types present in this graph.
An index of source type ID and their description follows:
%		 0: input scalar
%		 1: ALP/GraphBLAS container
%		 2: input iterator
%		 3: input integer
There are 13 unique operation vertex types present in this graph.
An index of vertex type ID and their description follows:
%		 6: clear( vector )
%		 7: setElement( vector )
%		 9: set( vector, scalar )
%		 12: set( vector, vector )
%		 50: mxm( matrix, matrix, matrix, semiring, scalar )
%		 51: mxm( matrix, matrix, matrix, monoid, scalar, scalar )
%		 52: outer( matrix, vector, vector, scalar, scalar )
%		 65: vxm( vector, vector, matrix, ring )
%		 69: mxv( vector, matrix, vector, ring )
%		 71: buildMatrixUnique( matrix, scalar, scalar, scalar )
%		 75: resize( matrix, scalar )
%		 81: vxm( vector, vector, matrix, scalar, scalar )
%		 86: foldl( vector, vector, monoid)
%There are 1 unique output vertex types present in this graph. 
An index of output vertex type ID and their description follows:
%		 103: output container

```
These gives us a systematic way of defining the type of a vertex (source, operation, output)

After the comments follow the so-called header line:

```
80 87 94
```
This indicates that there 80 hyperedges, 87 vertices, and 94 pins in the 
output HyperDAG. 

Then:

```
0 % no additional data on hyperedges at present
1 % no additional data on hyperedges at present
2 % no additional data on hyperedges at present
.
.
79 % no additional data on hyperedges at present

```

It gives information about every hyperedge in the hyperDag.
In this example we do not have any further informaton about the
hyperedges.

Then:
```
0 1 0 % source vertex of type ALP/GraphBLAS container no. 0
1 1 1 % source vertex of type ALP/GraphBLAS container no. 1
2 1 2 % source vertex of type ALP/GraphBLAS container no. 2
3 3 0 % source vertex of type input integer no. 0
.
.
.
85 103 0 % output vertex of type output container no. 0

```

Each libe has three numbers: a b c % ....    
which represent the following infomation  
a: is a serial number  (used to distinguish the different vertices)
b: the type of the vertex    
c: how many times this category of the vertex appears  

for example the third line:   

```
2 1 2 % source vertex of type ALP/GraphBLAS container no. 2

```

means that the third element that we register (we start counting
from zero) is a ALP/GraphBLAS container (as we registered in the 
begginning) and it has been appeared 3 times, until that time.

What then follows is:

```
0 0
0 4
1 1
1 6
2 2
2 77
2 79

```
a line for each of the pins, printed as a pair of hypergraph and vertex IDs.


Contents
========

A list of ALP/GraphBLAS algorithms from which hyperDAGs are extracted and
included in this database follows:
1. Conjugate Gradient, a common linear solver for SPD matrices;
2. BiCGstab, a common linear solver for general (invertible) matrices;
3. k-NN, a k-hop reachability query using matrix powers over a Boolean
   semiring;
4. PageRank, the canonical node ranking scheme by Brin and Page;
5. K-Means, the classical machine learning algorithm for clustering;
6. Label Propagation, the classical semi-supervised machine learning algorithm;
7. Sparse neural network inference (SNNI), based on the related IEEE/MIT
   GraphChallenge using RadixNet and the MNIST handwritten digits dataset;
8. Connected components, implemented using the ALP/Pregel API;
9. Any ALP/Pregel algorithm always exhibits the same computational structure,
   since they expand into standard ALP/GraphBLAS in a systematic fashion.

The HPCG, k-means, and label propagation tests built by ALP/GraphBLAS include
embedded datasets the algorithms are executed over. For all other algorithms,
an input dataset is required. For k-NN specifically, the parameter k is user
input as well. Thus, two flavours of hyperDAGs are included:

1. `./until_convergence`: here, the above algorithms are run until completion,
using a specific dataset and a specific parameter (e.g., k) when applicable. In
this directory, the dataset and parameters used to generate the hyperDAG are
noted within its file name.

2. `./limited_iterations`: here, algorithms 1, 3, and 5 from the above list are
executed for a limited number of iterations so that the hyperDAG captures the
lead-up, stable iteration state, and the wind-down of each algorithm. These
hyperDAGs are independent of the input dataset used to generate them, and
capture all phases within the algorithm they model.

ALP/Pregel
----------

For readers unfamiliar with Pregel, a short summary is included. Introduced by
Malewicz et al. in 2010, Pregel is a Google framework for graph computing that
introduced the concept of vertex-centric programming. Here, programmers write
algorithms that are run by vertices on a graph. The algorithm is executed
round-by-round, alternating between local computations, broadcasting messages
along the edges the vertex connects to, and aggregating incoming messages
before the next round of computation proceeds.

ALP/Pregel compile-time extends vertex-centric programs into standard
ALP/GraphBLAS programs, which, in turn, enables capturing hyperDAGs from such
programs. We capture two types of ALP/Pregel hyperDAGs:

1. in `./until_convergence`, the database includes a trace from a vertex-centric
connected components algorithm. This algorithm terminated at a certain number of
rounds that depend on the input dataset. The number of rounds taken is
documented in the comments, while the dataset used is noted in the file name.

2. in `./limited_iterations`, we make use of the round-based structure of Pregel
programs. We capture the lead-up of initialising the ALP/Pregel system on any
given input matrix, the steady-state of executing two rounds of any
vertex-centric program, and the wind-down of terminating the ALP/Pregel program.
This hyperDAG hence is independent of the input dataset as well as independent
of which vertex-centric program is modelled, as long as executing the program
would take more than one round.

Notes
-----

1. The canonical PageRank algorithm is available in ALP/GraphBLAS under thes
   name `simple_pagerank`. The related files in this database are named
   similarly.

