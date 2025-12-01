# 1.Set up the environment

To run this code, you need the following dependencies:
- Python 3.9.19
- ecole 0.7.2
- pytorch 2.5.1


# 2.Prepare data


## 2.1 Generate MILP instances

```
python -u 1_generate_instances.py setcover 
python -u 1_generate_instances.py cauctions 
python -u 1_generate_instances.py indset
python -u 1_generate_instances.py ufacilities
```

## 2.2 Generate supervised learning datasets

```
python -u 2_generate_il_samples.py setcover -j 8 
python -u 2_generate_il_samples.py cauctions -j 8
python -u 2_generate_il_samples.py indset -j 8
python -u 2_generate_il_samples.py ufacilities -j 8
```

# 3.Train the model

```
python -u 3_train_il_DA.py setcover -g 0
python -u 3_train_il_DA.py cauctions -g 0
python -u 3_train_il_DA.py indset -g 0
python -u 3_train_il_DA.py ufacilities -g 0
```
# 4.Evaluation

```
python -u 4_evaluate_DA.py setcover -g 0 
python -u 4_evaluate_DA.py cauctions -g 0 
python -u 4_evaluate_DA.py indset -g 0
python -u 4_evaluate_DA.py ufacilities -g 0 
```
