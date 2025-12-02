# This script is used for testing the model.

import numpy as np
from pandas import read_csv
import argparse
import os
import torch
from models_DA import Dual_Attention
from datetime import datetime
import pandas as pd

# Arguments of the script
parser = argparse.ArgumentParser()
parser.add_argument("--exp_env", default='2', choices=['1', '2'])
parser.add_argument("--data", help="number of testing data", default=1000)
parser.add_argument(
    "--set",
    help="which set you want to test on?",
    default="train",
    choices=['test', 'train']
)
parser.add_argument(
    "--type",
    help="what's the type of the model",
    default="fea",
    choices=['fea', 'obj']
)
parser.add_argument("--gpu", help="gpu index", default="0")
parser.add_argument("--model_key", default=None)
args = parser.parse_args()

def process(model, batch_data, type='fea'):

    c, ei, ev, v, cand_scores = batch_data
    
    model.eval()
    with torch.no_grad():
        logits = model(c, ei, ev, v)
        
        logits = logits.view(-1)
        cand_scores = cand_scores.view(-1)
    
    if type == "fea":
        # For feasibility: compute classification error rate
        logits_np = logits.detach().cpu().numpy()
        targets_np = cand_scores.detach().cpu().numpy()
        # False positives: predicted feasible but actually infeasible
        errs_fp = np.sum((logits_np > 0.5) & (targets_np < 0.5))
        # False negatives: predicted infeasible but actually feasible
        errs_fn = np.sum((logits_np < 0.5) & (targets_np > 0.5))
        errs = errs_fp + errs_fn
        return_err = errs / cand_scores.shape[0]
    else:
        # For objective value: compute MSE loss
        loss = torch.nn.functional.mse_loss(logits, cand_scores)
        return_err = loss.detach().cpu().numpy()
    
    return return_err

# SET-UP: Parse model configuration from model filename
temp = args.model_key.split('-')
type = temp[-1]
if args.set == "test" and args.exp_env == "2":
    temp[2] = "testing"
datafolder = temp[0]+'-'+temp[1]+'/'+'-'.join(s for s in temp[2:-1])
n_Samples_test = int(args.data)
n_Cons_small = 6
n_Vars_small = 20
n_Eles_small = 60 if "data-env1-unfoldable" in args.model_key else 12
# Collect all models matching the specified key
exp_list = []
for model_name in os.listdir("./saved-models/DA"):
    if args.model_key not in model_name:
        continue
    model_path = "./saved-models/DA/" + model_name
    # Extract embedding size from model filename
    embSize = int(model_name[:-4].split('-')[-1][1:])
    # Extract number of samples used for training
    n_Samples = (
        int(model_name.split('-')[-2][1:])
        if args.set == "train" else n_Samples_test
    )
    exp_list.append((model_path, embSize, n_Samples))

# Load different data files based on model type
if type == "fea":
    # For feasibility prediction: use all instances (feasible and infeasible)
    varFeatures_np = read_csv(
        datafolder + "/VarFeatures_all.csv", header=None
    ).values
    conFeatures_np = read_csv(
        datafolder + "/ConFeatures_all.csv", header=None
    ).values
    edgFeatures_np = read_csv(
        datafolder + "/EdgeFeatures_all.csv", header=None
    ).values
    edgIndices_np = read_csv(
        datafolder + "/EdgeIndices_all.csv", header=None
    ).values
    labels_np = read_csv(
        datafolder + "/Labels_feas.csv", header=None
    ).values
if type == "obj":
    # For objective prediction: use only feasible instances
    varFeatures_np = read_csv(
        datafolder + "/VarFeatures_feas.csv", header=None
    ).values
    conFeatures_np = read_csv(
        datafolder + "/ConFeatures_feas.csv", header=None
    ).values
    edgFeatures_np = read_csv(
        datafolder + "/EdgeFeatures_feas.csv", header=None
    ).values
    edgIndices_np = read_csv(
        datafolder + "/EdgeIndices_feas.csv", header=None
    ).values
    labels_np = read_csv(
        datafolder + "/Labels_obj.csv", header=None
    ).values

# Set-up device
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

# Prepare output file for results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if "unfoldable" in args.model_key:
    dataset_type = "unfoldable"
elif "rand" in datafolder or "randFeat" in datafolder:
    dataset_type = "foldable_rand"
else:
    dataset_type = "foldable"

output_file = f"{timestamp}_{type}_{dataset_type}_test_results.csv"
output_dir = f"./results/DA/{type}"
os.makedirs(output_dir,exist_ok=True)
results = []

def create_batch_data(conFeatures, edgIndices, edgFeatures, varFeatures, labels, 
                     n_samples, n_vars_small, n_cons_small, n_eles_small, batch_size=1000):
    batch_list = []
    
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        current_batch_size = batch_end - batch_start
        
        # Collect data for current batch
        batch_cons = []
        batch_vars = []
        batch_edges = []
        batch_edge_indices = []
        batch_labels = []
        
        for i in range(batch_start, batch_end):
            # Extract data ranges for single instance
            var_start, var_end = i * n_vars_small, (i + 1) * n_vars_small
            con_start, con_end = i * n_cons_small, (i + 1) * n_cons_small  
            edge_start, edge_end = i * n_eles_small, (i + 1) * n_eles_small
            
            # Collect single graph data
            batch_cons.append(conFeatures[con_start:con_end])
            batch_vars.append(varFeatures[var_start:var_end])
            batch_edges.append(edgFeatures[edge_start:edge_end])
            
            # Adjust edge indices to be relative to single graph
            sample_edge_indices = edgIndices[:, edge_start:edge_end].clone()
            sample_edge_indices[0] -= i * n_cons_small
            sample_edge_indices[1] -= i * n_vars_small 
            batch_edge_indices.append(sample_edge_indices)
            
            # Collect labels
            batch_labels.append(labels[i])
        
        # Convert lists to tensors, maintaining batch dimension
        batch_cons_tensor = torch.stack(batch_cons, dim=0)
        batch_vars_tensor = torch.stack(batch_vars, dim=0) 
        batch_edges_tensor = torch.stack(batch_edges, dim=0)
        batch_edge_indices_tensor = torch.stack(batch_edge_indices, dim=0)
        
        batch_labels_tensor = torch.stack(batch_labels, dim=0)
        
        batch_item = (
            batch_cons_tensor,
            batch_edge_indices_tensor,
            batch_edges_tensor,
            batch_vars_tensor,
            batch_labels_tensor
        )
        batch_list.append(batch_item)
    
    return batch_list

# Main testing loop
for model_path, embSize, n_Samples in exp_list:

    # Verify actual data availability
    max_samples_from_con = conFeatures_np.shape[0] // n_Cons_small
    max_samples_from_var = varFeatures_np.shape[0] // n_Vars_small  
    max_samples_from_edge = edgFeatures_np.shape[0] // n_Eles_small

    max_samples_from_labels = labels_np.shape[0]

    # Use minimum to ensure data consistency
    actual_available_samples = min(max_samples_from_con, max_samples_from_var, 
                                max_samples_from_edge, max_samples_from_labels)
    n_Samples =  min(n_Samples, actual_available_samples)

    # Load dataset into gpu
    varFeatures = torch.tensor(
        varFeatures_np[:n_Vars_small * n_Samples, :], dtype=torch.float32
    ).to(device)
    conFeatures = torch.tensor(
        conFeatures_np[:n_Cons_small * n_Samples, :], dtype=torch.float32
    ).to(device)
    edgFeatures = torch.tensor(
        edgFeatures_np[:n_Eles_small * n_Samples, :], dtype=torch.float32
    ).to(device)
    edgIndices = torch.tensor(
        edgIndices_np[:n_Eles_small * n_Samples, :], dtype=torch.long
    ).to(device)
    edgIndices = torch.transpose(edgIndices, 0, 1)


    labels = torch.tensor(labels_np[:n_Samples, :], dtype=torch.float32).to(device)

    # Get feature dimensions
    nConsF = conFeatures.shape[1]
    nVarF = varFeatures.shape[1]
    nEdgeF = edgFeatures.shape[1]
    n_Cons = conFeatures.shape[0]
    n_Vars = varFeatures.shape[0]

    # Create batched data for evaluation
    batch_data_list = create_batch_data(conFeatures, edgIndices, edgFeatures, varFeatures, labels,
                                   n_Samples, n_Vars_small, n_Cons_small, n_Eles_small, 
                                   batch_size=32)
    # Set Dual-Attention hyperparameters
    d_model = embSize   
    d_ff = embSize * 2     
    d_k = d_v = embSize  
    n_layers = 4    
    n_heads = 2     

    model = Dual_Attention(embSize, nConsF, nEdgeF, nVarF,
                        n_layers, n_heads, d_model, d_k, d_v, d_ff,
                        isGraphLevel=True)
    model = model.to(device)

    # Load trained model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        continue

    total_err = 0
    total_samples = 0
    for batch_data in batch_data_list:
        batch_err = process(model, batch_data, type=type)
        batch_size = batch_data[0].shape[0]
        total_err += batch_err * batch_size
        total_samples += batch_size

    err = total_err / total_samples

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"MODEL: {model_path}, DATA-SET: {datafolder}, "
        f"NUM-DATA: {n_Samples}, EXP: {args.exp_env}, ERR: {err},"
        f"PARAMS: {total_params:,}"
    )

    # Store results for CSV output
    result = {
        'model_path': os.path.basename(model_path),
        'embedding_size': embSize,
        'num_parameters': total_params,
        'error_rate': err,
        'num_samples': n_Samples
    }
    results.append(result)

# Save results to csv and text files
if results:
    df = pd.DataFrame(results)
    df = df.sort_values('embedding_size')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, output_file)
    df.to_csv(output_path, index=False)
    output_path = output_path.replace('csv','txt')
    with open(output_path, 'w') as f:
        f.write(df.to_string(index=False))

    print(f"\nResults saved to: {output_path}")
    print(f"Total models tested: {len(results)}")