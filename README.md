Query-free Black-box Adversarial Attacks on Graphs
===============================================================================

About
-----

This project is the implementation of the paper "Query-free Black-box Topology Attacks in Graph-based Learning".
A query-free black-box adversarial attack on graphs is proposed, where the attacker has no knowledge of the target model and no query access to the model. With the mere observation of the graph topology, the proposed attack strategy aim to flip a limited number of links to mislead the graph model.

This repo contains the codes, data and results reported in the paper.

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


Usage
-----
We focus on the query-free black-box attack on graphs, in which the attacker could only observe the input graph, but has no knowledge about the victim model and can not query any examples.

### Input Format
Following our settings, we only need the structure information of input graphs to perform our attacks.
An example data format is given in ```data``` where dataset is in ```npz``` format.

When using your own dataset, you must provide:

* an N by N adjacency matrix (N is the number of nodes).

### Output Format
The program outputs to a file in ```npz``` format which contains the adversarial edges.

### Main Script
The help information of the main script ```attack.py``` is listed as follows:

    python attack.py -h
    
    usage: attack.py [-h][--dataset] [--pert-rate] [--threshold] [--save-dir]
    
    optional arguments:
      -h, --help                Show this help message and exit
      --dataset                 str, The dataset to be perturbed on [cora, citeseer, polblogs].
      --pert-rate               float, Perturbation rate of edges to be flipped.
      --threshold               float, Restart threshold of eigen-solutions.
      --save-dir                str, File directory to save outputs.
      
### Demo
We include all three benchmark datasets Cora-ML, Citeseer and Polblogs in the ```data``` directory.
Then a demo script is available by calling ```attack.py```, as the following:

    python attack.py --data-name cora --pert-rate 0.1 --threshold 0.03 
    
    
Evaluations
-----
Our evaluations depend on the output adversarial edges by the above attack model.
We provide the evaluation codes of our attack strategy on the node classification task here. 
We evaluate on three real-world datasets Cora-ML, Citeseer and Polblogs. 
Our setting is the poisoning attack, where the target models are retrained after perturbations.
We use GCN, Node2vec and Label Propagation as the target models to attack.

<!--
We provide the evaluation codes of node-level attack and graph-level attack here. 

### Node-level Attack
For node-level attack, we perform our attack strategy to the node classification task. 
We evaluate on three real-world datasets Cora-ML, Citeseer and Polblogs. 
Our setting is the poisoning attack, where the target models are retrained after perturbations.
We use GCN, Node2vec and Label Propagation as the target models to attack.
If you want to attack GCN, you can run ```evaluation/eval_gcn.py```.
If you want to attack Node2vec, you can run ```evaluation/eval_emb.py```.
If you want to attack Label Propagation, you can run ```evaluation/eval_lp.py```.
     
### Graph-level Attack
For graph-level attack, we perform our attack strategy to the graph classification task. 
We evaluate on two protein datasets: Enzymes and Proteins. 
We use GIN and Diffpool as our target models to attack.
If you want to attack GIN, you can run ```evaluation/eval_gin.py```.
If you want to attack Diffpool, you can run ```evaluation/eval_diffpool.py```.
-->

### Evaluation Script

If you want to attack GCN, you can run ```evaluation/eval_gcn.py```.
The help information of the evaluation script is listed as follows:

    python . -h
    
    usage: . [-h][--dataset] [--pert-rate] [--dimensions] [--load-dir]
    
    optional arguments:
      -h, --help                Show this help message and exit
      --dataset                 str, The dataset to be evluated on [cora, citeseer, polblogs].
      --pert-rate               float, Perturbation rate of edges to be flipped.
      --dimensions              str, Dimensions of GCN hidden layer. Default is 16.
      --load-dir                str, File directory to load adversarial edges.
       
       
If you want to attack Label Propagation, you can run ```evaluation/eval_emb.py```.
The help information of the evaluation script is listed as follows:

    python . -h
    
    usage: . [-h][--dataset] [--pert-rate] [--dimensions] [--window-size] [--load-dir]
    
    optional arguments:
      -h, --help                Show this help message and exit
      --dataset                 str, The dataset to be evluated on [cora, citeseer, polblogs].
      --pert-rate               float, Perturbation rate of edges to be flipped.
      --dimensions              int, Output embedding dimensions of Node2vec. Default is 32.
      --window-size             int, Context size for optimization in Node2vec. Default is 5.
      --walk-length             int, Length of walk per source in Node2vec. Default is 80.
      --walk-num                int, Number of walks per source in Node2vec. Default is 10.
      --p                       float, Parameter in node2vec. Default is 4.0.
      --q                       float, Parameter in node2vec. Default is 1.0.
      --worker                  int, Number of parallel workers. Default is 10.
      --load-dir                str, File directory to load adversarial edges.
      
If you want to attack Node2vec, you can run ```evaluation/eval_lp.py```.
The help information of the evaluation script is listed as follows:

    python . -h
    
    usage: . [-h][--dataset] [--pert-rate] [--load-dir]
    
    optional arguments:
      -h, --help                Show this help message and exit
      --dataset                 str, The dataset to be evluated on [cora, citeseer, polblogs].
      --pert-rate               float, Perturbation rate of edges to be flipped.
      --load-dir                str, File directory to load adversarial edges.
