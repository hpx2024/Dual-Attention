# 1. Set Up the Environment

To run this project, you will need the following dependencies:

- Python 3.9.19
- pyscipopt 3.0.4
- ecole 0.7.2
- pytorch 2.5.1

Follow the steps below to create and configure the environment:

```bash
cd 
conda create -n DAB-Solving-State python=3.9.19
conda activate DAB-Solving-State

pip install pyscipopt==3.0.4
pip install cython
pip install wheel
pip install numpy
pip install scikit-build

git clone git@github.com:lascavana/ecole.git
cd ecole
mkdir wheels
python setup.py bdist_wheel --dist-dir wheels
pip install --no-index --find-links=wheels ecole

conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
TORCH=2.5.1
CUDA=cu121
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric==2.6.1

conda install -c conda-forge fmt=11
conda install -c conda-forge tbb=2020
```


# 2. Prepare Data
This section describes how to generate both MILP instances and imitation learning datasets used for training the model.

## 2.1 Generate MILP Instances

Run the following commands to generate the four benchmark MILP intances:

```bash
python -u 1_generate_instances.py setcover 
python -u 1_generate_instances.py cauctions 
python -u 1_generate_instances.py indset
python -u 1_generate_instances.py ufacilities
```

## 2.2 Generate Imitation Learning Datasets

After the MILP instances are created, generate the IL training samples:

```bash
python -u 2_generate_il_samples.py setcover -j 8 
python -u 2_generate_il_samples.py cauctions -j 8
python -u 2_generate_il_samples.py indset -j 8
python -u 2_generate_il_samples.py ufacilities -j 8
```

# 3. Train the Model

Run the following commands to train the model for each problem type:

```bash
python -u 3_train_il_DA.py setcover -g 0
python -u 3_train_il_DA.py cauctions -g 0  
python -u 3_train_il_DA.py indset -g 0
python -u 3_train_il_DA.py ufacilities -g 0
```
# 4. Evaluation

After training, evaluate the learned models on the test sets using the following commands:

```bash
python -u 4_evaluate_DA.py setcover -g 0 
python -u 4_evaluate_DA.py cauctions -g 0 
python -u 4_evaluate_DA.py indset -g 0
python -u 4_evaluate_DA.py ufacilities -g 0 
```
