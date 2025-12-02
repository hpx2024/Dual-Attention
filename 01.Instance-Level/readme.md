# Environment Setup
To run this code, you need the following dependencies:
- Python 3.12.3
- pytorch 2.5.1
- pandas 2.3.2

You can refer to the following methods to create the current code environment
```
conda create -n DAB-Intance
conda activate DAB-Intance
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::pandas
conda install conda-forge::pyscipopt
conda install conda-forge::tqdm
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

cd 01.Instance-Level
```


# Reproduce steps

## Step 1: Generate data

```
python 1_generate_data.py --exp_env 1
python 1_generate_data.py --exp_env 2
```

## Step 2: Training a model for the feasibility or  of MILP.

```
python 2_training_DA.py --type fea --data 1000 --epoch 10000 --data_path data-env1/foldable-randFeat --embSize 64
python 3_testing_DA.py --exp_env 2 --model_key data-env1-foldable-randFeat-fea
```

## Step 3: Training and testing a GNN for the Objective value of MILP.

```
python 2_training_DA.py --type obj --data 1000 --epoch 12000 --data_path data-env1/unfoldable --embSize 64
python 3_testing_DA.py --exp_env 2 --model_key data-env1-foldable-randFeat-obj --type obj
```


