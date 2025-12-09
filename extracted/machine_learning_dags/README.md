# DAG Artifacts from Hugging Face Transformers

This directory contains Directed Acyclic Graphs (DAGs) extracted from various Hugging Face Transformers models. The graphs are produced using `torch.compile` and exported as `.dot` files for inspection and analysis.

## Contents and Structure

- The DAGs represent the computation graphs generated during model compilation with PyTorch.
- Due to graph breaks introduced during tracing or compilation, some models result in multiple separate `.dot` files.
- Only the largest and most informative graph segment for each model is included in this directory.

## What the DOT Files Contain

- Each node in a `.dot` file corresponds to an operation emitted during compilation.
- The operation name and its tensor shape are recorded directly in the graph.
- Shapes refer to the tensor dimensions; to estimate the memory footprint of an operation, multiply the number of elements by the precision size  
  (for example, half precision uses 16 bits or 2 bytes per element).

## CPU Profiling Metrics

- Some models include `cpu-time` and `cpu-mem` fields extracted using `torch.profiler`.
- These values represent approximate compute and memory usage for different operations.
- A direct correspondence between profiler events and operations in the `.dot` graphs is often difficult to establish due to differences in tracing and profiling granularity.
- For this reason, these metrics should be considered complementary reference information rather than exact per-node measurements.

## Notes

- DOT files can be visualized using Graphviz or compatible tools to inspect the model execution flow.
