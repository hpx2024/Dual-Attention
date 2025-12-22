# 1. Prepare the Dataset

## 1.1 Prepare Instances
The **IP** and **WA** datasets come from the **ML4CO 2021 competition**, containing:

- **Training set:** 400 instances (`0`–`399`)
- **Test set:** 100 instances (`9900`–`9999`)

Download link: https://github.com/ds4dm/ml4co-competition/blob/main/DATA.md

The **CA** and **IS** datasets are generated using the **Ecole** library. These instances are specifically constructed to be difficult for the Gurobi solver.

Before generating CA and IS instances, you must create a dedicated environment for the Ecole library:

```bash
conda create -n ecole_env python=3.9
conda activate ecole_env

conda install -c conda-forge ecole
conda install -c conda-forge ecole pyscipopt
```

Then generate the CA and IS datasets:

```bash
python 1_generate_insteances.py
```

If you encounter the error:
ImportError: libscip.so.8.0: cannot open shared object file: No such file or directory
install SCIP 8.0.4 explicitly:

```bash
conda install scip=8.0.4
```

## 1.2 Environment for the Main Code

The remaining scripts require an environment with the following dependencies:
- Python 3.12.2
- gurobipy 12.0.1
- pytorch 2.4.1

You can create the environment as follows (Gurobi must be activated):

```bash
conda create -n DAB-Element python=3.12.2
conda activate DAB-Element

conda install gurobi::gurobi==12.0.1
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge pyscipopt=5.0.1
conda install conda-forge::pytorch_geometric
conda install conda-forge::matplotlib
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cu121.html

```



## 1.3 Collect Solutions

This step collects feasible solutions for each training instance, which will later be used for training the model.

```bash
python utils/collect_sols.py --dataDir ./datasets/IP/train --nWorkers 5 --maxTime 3600 --maxStoredSol 50
python utils/collect_sols.py --dataDir ./datasets/IS/train --nWorkers 5 --maxTime 3600 --maxStoredSol 50
python utils/collect_sols.py --dataDir ./datasets/CA/train --nWorkers 5 --maxTime 3600 --maxStoredSol 50
python utils/collect_sols.py --dataDir ./datasets/WA/train --nWorkers 5 --maxTime 3600 --maxStoredSol 50
```


## 1.4 Collect Raw Features

This section extracts the raw features of instance, which will be used as input for the model.

```bash
python utils/collect_bg.py --dataDir ./datasets/IP/train
python utils/collect_bg.py --dataDir ./datasets/WA/train
python utils/collect_bg.py --dataDir ./datasets/IS/train
python utils/collect_bg.py --dataDir ./datasets/CA/train
```


# 2. Train the Model

Train the model separately on each dataset.

```bash
python 2_train.py --dataset IP
python 2_train.py --dataset WA
python 2_train.py --dataset CA
python 2_train.py --dataset IS
```

# 3. Downstream Tasks

The following steps evaluate the trained models on different downstream tasks.

## 3.1 Collect Predicted Solutions

Use the trained models to generate predictions for each dataset. These predictions will be used to guide solvers.

```bash
python utils/get_logits.py --dataset IP
python utils/get_logits.py --dataset WA
python utils/get_logits.py --dataset CA
python utils/get_logits.py --dataset IS
```

## 3.2 Task-PaS

In the PaS task, the model predictions are used to guide Gurobi for more efficient Local brancing search. Multiple threads can be used for acceleration:

```bash
python 3_1eval_ps_GRB.py --dataset IP --nWorkers 8
python 3_1eval_ps_GRB.py --dataset WA --nWorkers 8
python 3_1eval_ps_GRB.py --dataset CA --nWorkers 8
python 3_1eval_ps_GRB.py --dataset IS --nWorkers 8
```

## 3.3 Task-Apollo

Apollo uses a solver to correct model predictions and guide solver solving, which is a further work of PaS.

```bash
python 3_2eval_Apollo_GRB.py -p WA -t 8
python 3_2eval_Apollo_GRB.py -p IP -t 8
python 3_2eval_Apollo_GRB.py -p CA -t 8
python 3_2eval_Apollo_GRB.py -p IS -t 8
```
