# 1.Prepare the dataset
## 1.1 Prepare instances
The IP and WA datasets are from the ML4CO 2021 competition, with 400 files numbered from 0 to 399 in the training set and 100 files numbered from 9900 to 9999 in the testing set.
Download link: https://github.com/ds4dm/ml4co-competition/blob/main/DATA.md

CA and IS datasets utilize the ecole library (encapsulating Gasse source code) to generate instances that are difficult for Gurobi solvers to solve.

```
python 1_generate_insteances.py
```

To execute the above code, you need to create an additional environment for the Ecole library

```
conda create -n ecole_env python=3.9
conda activate ecole_env

conda install -c conda-forge ecole
conda install -c conda-forge ecole pyscipopt
```

If you run the code with the error message 'ImportError: libscip. so. 8.0: cannot open shared object file: No such file or directory', you need to run the following code to restrict scip to version 8.0.4
```
conda install scip=8.0.4
```


For the remaining codes, you need to run them in an environment with the following dependencies:
- Python 3.12.2
- gurobipy 12.0.1
- pytorch 2.4.1

```
conda create -n DAB-Element python=3.12
conda activate 
```



## 1.2 Collect solutions


```
python utils/collect_sols.py --dataDir ./datasets/IP/train --nWorkers 5 --maxTime 3600 --maxStoredSol 50
python utils/collect_sols.py --dataDir ./datasets/IS/train --nWorkers 5 --maxTime 3600 --maxStoredSol 50
python utils/collect_sols.py --dataDir ./datasets/CA/train --nWorkers 5 --maxTime 3600 --maxStoredSol 50
python utils/collect_sols.py --dataDir ./datasets/WA/train --nWorkers 5 --maxTime 3600 --maxStoredSol 50
```


## 1.3 Collect bipartite graph


```
python utils/collect_bg.py --taskName PS --dataDir ./datasets/IP/train
python utils/collect_bg.py --taskName PS --dataDir ./datasets/WA/train
python utils/collect_bg.py --taskName PS --dataDir ./datasets/IS/train
python utils/collect_bg.py --taskName PS --dataDir ./datasets/CA/train
```


# 2.Train the model

```
python train_ps.py --dataset IP
python train_ps.py --dataset WA
python train_ps.py --dataset CA
python train_ps.py --dataset IS
```

# 3.Downstream tasks

## 3.1 Collect predicted solutions

```
python utils/get_logits_ps.py --dataset IP
python utils/get_logits_ps.py --dataset WA
python utils/get_logits_ps.py --dataset CA
python utils/get_logits_ps.py --dataset IS
```

## 3.2 Task-PaS

```
python 3_1eval_ps_GRB.py --dataset IP --nWorkers 8
python 3_1eval_ps_GRB.py --dataset WA --nWorkers 8
python 3_1eval_ps_GRB.py --dataset CA --nWorkers 8
python 3_1eval_ps_GRB.py --dataset IS --nWorkers 8
```

## 3.3 Task-Apollo

```
python eval_ps_Apollo_GRB.py -p WA -t 8
python eval_ps_Apollo_GRB.py -p IP -t 8
python eval_ps_Apollo_GRB.py -p CA -t 8
python eval_ps_Apollo_GRB.py -p IS -t 8
```
