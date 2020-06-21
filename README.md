Query-free Black-box Adversarial Attacks on Graphs
===============================================================================

About
-----

This project is the implementation of the paper "Query-free Black-box Topology Attacks in Graph-based Learning".

Dependencies
-----

The script has been tested running under Python 3.7.7, with the following packages installed (along with their dependencies):

- `numpy==1.18.1`
- `scipy==1.4.1`
- `scikit-learn==0.23.1`
- `gensim==3.8.0`
- `networkx==2.3`
- `tqdm==4.46.1`
- `torch==1.4.1`
- `torch_geometric==1.5.0`


Some Python module dependencies are listed in `requirements.txt`, which can be easily installed with pip:

```
pip install -r requirements.txt
```

In addition, CUDA 10.0 has been used in our project. Although not all dependencies are mentioned in the installation instruction links above, you can find most of the libraries in the package repository of a regular Linux distribution.


Input Format
-----
An example data format is given in ```data``` where dataset is in ```npz``` format.

When using your own dataset, you must provide:

* an N by N adjacency matrix (N is the number of nodes).
* an N by C one-hot label matrix (C is the number of classes).
* (OPTIONAL) an N by D feature matrix (D is the dimension of node features).

How to run 
-----
The help information of the main script ```attack.py``` is listed as follows:

    python attack.py -h
    
    usage: attack.py [-h][--dataset] [--pert-rate] [--threshold] 
    
    optional arguments:
      -h, --help                show this help message and exit
      --dataset                 str, the dataset to be perturbed on [cora, citeseer, polblogs].
      --pert-rate               float, perturbation rate of edges to be flipped.
      --threshold               float, restart threshold of eigen-solutions.
      

To reproduce the results that reported in the paper, you can run the following command:

    python attack.py --data-name cora --pert-rate 0.1 --threshold 0.03 
