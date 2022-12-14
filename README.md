
This repository, the HyperDAGs Database and its tools, is copyright by the
Computing Systems Laboratory, Zurich Research Center, Huawei Technologies
Switzerland AG.

Data and tools are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the applicable licenses
for the specific language governing permissions and limitations.

# HyperDAGs

HyperDAGs contained in this database are licensed under the Creative Commons
Attribution 4.0 International License. To view a copy of this license, visit

   http://creativecommons.org/licenses/by/4.0/

or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

The hyperDAG file format is described [here](FileFormat_README.md). HyperDAGs
in this database are organised across the following directories:

## ./extracted

These correspond to HyperDAGs that are automatically and systematically
extracted programming frameworks such as ALP. These are, unless otherwise noted,
coarse-grained HyperDAGs. These are tied to specific algorithms implemented from
such a framework.

For iterative methods, the HyperDAGs herein contained presently are a mixture of
generic iterative structures (encoding both its lead-up, its stable-state, and
its coda), as well as the full coarse-grained structure that arises from running
iterative methods to convergence. In the latter case, the HyperDAG is tied to
specific input data.

## ./fine-grained

These contain fine-grained HyperDAGs that are not extracted from practical
frameworks, typically tied to a specific algorithm on specific input data.

## ./synthetic

These contain synthetic HyperDAGs, typically generated by a specifically-
designed tool using varying random processes.

# Tools

Tools in this repository are found in the ./tools directory. They include random
hyperDAG generators, hyperDAG visualisation scripts, as well as converter tools
between DAGs and HyperDAGs.

All tools in this directory are licensed under the Apache License, Version 2.0
(the "License"); you may not use the tools except in compliance with the
License. You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

