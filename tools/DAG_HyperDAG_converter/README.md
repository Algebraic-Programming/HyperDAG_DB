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

-------------------------
DAG <--> HYPERDAG CONVERTER
-------------------------

This is a simple tool to convert a computational DAG into a HyperDAG and vice versa. It was created at Computational Systems Laboratory, Zurich Research Center, Huawei Technologies Switzerland.

Disclaimer: the tool is a research prototype, and as such, it might occasionally contain unexpected bugs or runtime errors. It also assumes that it is run with a correct parametrization: the correctness of the program parameters, and the correct format of the input file are only checked to a limited extent.

-------------------
PROGRAM PARAMETERS:
-------------------

"-input": required parameter, describes the name of the input file from which the computational DAG is read.

"-output": optional parameter, describes the name of the output file where the hyperDAG is written. If not specified, the default name is "out_hyperDAG.txt".

"-DAGtoHyperDAG": if set, the program interprets the input as a DAG file, and converts it into a hyperDAG file. You must use exactly one of "-DAGtoHyperDAG" and "-HyperDAGtoDAG". Note that parameter reading is case sensitive.

"-HyperDAGtoDAG": if set, the program interprets the input as a HyperDAG file, and converts it into a DAG file. You must use exactly one of "-DAGtoHyperDAG" and "-HyperDAGtoDAG". Note that parameter reading is case sensitive.

"-debugMode": optional parameter, enables debugging messages on the standard output. This parameter does not need a following argument.

Some example calls:  
-input infile.txt -DAGtoHyperDAG  
-input infile.txt -output outfile.txt -HyperDAGtoDAG  
-input infile.txt -debugMode -DAGtoHyperDAG  

----------------
DAG FILE FORMAT:
----------------

The description of a computational DAG is in the following format. In DAGtoHyperDAG mode, the input file must follow this format; in HyperDAGtoDAG mode, the output file will be created in this format.

1. The first few lines, starting with a '%' character, can be comment lines giving a brief description of the DAG. These lines are copied to the beginning of the output HyperDAG file, but otherwise ignored for the processing.

2. The next line (the first non-comment line) should contain two integers "N E", separated by a space: the number of nodes N and the number of directed edges E in the DAG in this order.

3. This should be followed by N distinct lines, each with two integers "V W", separated by a space. This describes that the work weight of node number V is W (where W is a non-negative integer). Each of the indices 0, 1, ..., N-1 should appear exactly once among these lines.
 
4. This should be followed by N distinct lines, each with two integers "V W", separated by a space. This describes that the communication weight of node number V is W (W is again a non-negative integer). Each of the indices 0, 1, ..., N-1 should appear exactly once among these lines. 
[We note that the communication weight of sink nodes in the DAG is not meaningful, it will not affect the output hyperDAG. We keep it in the format nonetheless, since having exactly N lines here singnificantly simplifies the file reading process.]

5. Finally, this should be followed by E distinct lines, each describing a directed edge of the DAG. Each of these lines contain two integers “V1 V2”, separated by a space, meaning that there is a directed edge from node V1 to node V2. Nodes are indexed from 0 to N-1. 

In case your DAG has uniform weights, you can simply write "i 1" in the i-th line of both point 3. and point 4. above, indexing i from 0 to N-1.

The file should not have any trailing empty or comment lines.

While it is not strictly required, we still heavily recommend to specify the input DAGs in a format where the indexing of nodes is a topological ordering of the DAG, i.e. all directed edges go from lower-indexed to higher-indexed nodes.

The file "sample_DAG.txt" provides an example for the format described above.

---------------------
HYPERDAG FILE FORMAT:
---------------------

The description of HyperDAGs is in the same format as the rest of the weighted HyperDAGs in the database. In HyperDAGtoDAG mode, the input file must follow this format; in DAGtoHyperDAG mode, the output file will be created in this format.

1. The first few lines, starting with a '%' character, can be comment lines giving a brief description of the HyperDAG. These lines are copied to the beginning of the output DAG file, but otherwise ignored for the processing.

2. The next line (the first non-comment line) should contain three integers "M N P", separated by a spaces: the number of hyperedges M, the number of nodes N, and the number of pins P in the hyperDAG in this order.

3. This should be followed by M distinct lines, each with two integers "H W", separated by a space, where H is a hyperedge index between 0 and M-1, and W is the weight associated with this hyperedge (its "communication cost").

4. This should be followed by N distinct lines, each with two integers "V W", separated by a space, where V is a node index between 0 and N-1, and W is the weight associated with this node (its "work cost").

5. Finally, this should be followed by P distinct lines, each describing a specific pin. Each of these lines must contain two integers “H V”, separated by a space: the first is the index of the hyperedge, the second is the index of the node. Once again, nodes are indexed from 0 to N-1, hyperedges are indexed from 0 to M-1.

The file should not have any trailing empty or comment lines.

The file "sample_HyperDAG.txt" provides an example for the format described above.

