# 1. Environment Setup
To run this code, you will need the following dependencies:
- Python 3.11.9
- pytorch 2.5.1
- pandas 2.3.2
- pyscipopt 4.4.0

You can create the environment using the commands below:
```
conda create -n DAB-Instance
conda activate DAB-Instance

conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::pandas
conda install conda-forge::pyscipopt
conda install conda-forge::tqdm
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

cd 01.Instance-Level
```

# 2. Generate Data

Use the following commands to generate training and testing datasets:

```bash
python 1_generate_data.py --exp_env 1
python 1_generate_data.py --exp_env 2
```

# 3. Train and Test

## 3.1 Train and Test the Model for MILP Feasibility

Train the model to predict feasibility on the training set and evaluate it on testing set:

```bash
python 2_training_DA.py --type fea --data 1000 --epoch 10000 --data_path data-env1/foldable-randFeat --embSize 64
python 3_testing_DA.py --exp_env 2 --model_key data-env1-foldable-randFeat-fea
```

## 3.2 Train and Test the Model for MILP Objective Value

Train the model for objective-value prediction on the training set and evaluate it on the test set:

```bash
python 2_training_DA.py --type obj --data 1000 --epoch 12000 --data_path data-env1/foldable-randFeat --embSize 64
python 3_testing_DA.py --exp_env 2 --model_key data-env1-foldable-randFeat-obj --type obj
```


