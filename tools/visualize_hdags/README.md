# Python Installation Instructions

Python 3 and pip required. Steps:
1. Install jupyter lab: `pip install jupyterlab`
2. Install required packages: `pip install -r requirements.txt`

# Using the notebook

Start notebook with the command
- `jupyter-lab`

In the top cells of the notebook, enter
- relative or absolute path of hyperdag file
- number of entries to read (1000 or fewer recommend, as visualisation is slow
  and unhelpful for larger hyperDAGs)

Then execute the remaining cells to get a DAG and hyperDAG visualisation of the
input file.

