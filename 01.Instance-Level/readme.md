# Environment Setup
To run this code, you need the following dependencies:
- Python 3.12.3
- pytorch 2.5.1
- pandas 2.3.2


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


