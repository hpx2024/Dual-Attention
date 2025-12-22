# This script is used for training MILP instances.

import numpy as np
from pandas import read_csv
import argparse
import os

import torch
import torch.optim as optim
from models_DA import Dual_Attention

# Argument of the script
parser = argparse.ArgumentParser()
parser.add_argument("--data", help="number of training data", default=1000)
parser.add_argument("--gpu", help="gpu index", default="0")
parser.add_argument("--embSize", help="embedding size of GNN", default="6")
parser.add_argument("--epoch", help="num of epoch", default="10000")
parser.add_argument(
    "--type",
    help="what's the type of the model",
    default="obj",
    choices=['fea', 'obj']
)  # fea: feasibility, obj: objective value
parser.add_argument("--data_path", 	help="path of data", default="data-env1/foldable-randFeat")
parser.add_argument("--accumulation_steps", help="gradient accumulation steps", default="1")
args = parser.parse_args()


# Function of Training per epoch
def process(model, dataloader, optimizer, type='fea'):

    # Training phase with gradient accumulation
    batch_list = dataloader
    model.train()
    optimizer.zero_grad()

    train_loss_sum = 0
    accumulation_steps = int(args.accumulation_steps)
    accumulation_count = 0

    for i, batch_data in enumerate(batch_list):
        c, ei, ev, v, cand_scores = batch_data
        
        # Forward pass
        logits = model(c, ei, ev, v)
        # Reshape logits and targets based on model type
        logits = logits.view(-1)
        cand_scores = cand_scores.view(-1)
        
        # Calculate loss and normalize by accumulation steps
        loss = torch.nn.functional.mse_loss(logits, cand_scores)
        original_loss = loss.item()
        if len(batch_list) < accumulation_steps:
            accumulation_steps = len(batch_list)
        loss = loss / accumulation_steps
        loss.backward()
        
        train_loss_sum += original_loss
        accumulation_count += 1
    
        # Update weights only after accumulating enough gradients
        if accumulation_count % accumulation_steps == 0 or i == len(batch_list) - 1:
            optimizer.step()
            optimizer.zero_grad()

    # Evaluation phase
    model.eval()
    eval_loss_sum = 0
    total_errs = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_data in batch_list:
            c, ei, ev, v, cand_scores = batch_data
            
            logits = model(c, ei, ev, v)
            # Reshape based on model type
            if type != 'sol':
                logits = logits.squeeze(-1)
                cand_scores = cand_scores.squeeze(-1)
            else:
                logits = logits.view(-1)
                cand_scores = cand_scores.view(-1)

            loss = torch.nn.functional.mse_loss(logits, cand_scores)
            eval_loss_sum += loss.item()
            
            # Calculate classification errors for feasibility prediction
            if type == "fea":
                logits_np = logits.detach().cpu().numpy()
                targets_np = cand_scores.detach().cpu().numpy()
                # False positives: predicted feasible but actually infeasible
                errs_fp = np.sum((logits_np > 0.5) & (targets_np < 0.5))
                # False negatives: predicted infeasible but actually feasible
                errs_fn = np.sum((logits_np < 0.5) & (targets_np > 0.5))
                total_errs += errs_fp + errs_fn
                total_samples += cand_scores.shape[0]

    # Calculate averages
    return_loss = eval_loss_sum / len(batch_list)
    errs = total_errs if type == "fea" else None
    err_rate = total_errs / total_samples if type == "fea" else None

    return return_loss, errs, err_rate

# Set-up hyper parameters
max_epochs = int(args.epoch)
lr = 0.0008
seed = 0

# Set-up dataset
trainfolder = args.data_path
n_Samples = int(args.data)
n_Cons_small = 6  # Number of constraints per MILP instance
n_Vars_small = 20  # Number of variables per MILP instance

# Determine number of nonzero elements based on dataset type
if trainfolder == "data-env1/unfoldable":
    n_Eles_small = 60
else:
    n_Eles_small = 12

# Set-up embsize
embSize = int(args.embSize)

# Create directories for saving models
if not os.path.exists('./saved-models/'):
    os.mkdir('./saved-models/')
if not os.path.exists('./saved-models/DA/'):
    os.mkdir('./saved-models/DA/')
model_setting = trainfolder.replace('/', '-')

# Create directories for saving models
model_path = (
    './saved-models/DA/' +
    model_setting +
    '-' + args.type +
    '-d' + str(n_Samples) +
    '-s' + str(embSize) +
    '.pth'
)

# Load dataset into memory
# Load different data files based on model type
if args.type == "fea":
    # For feasibility prediction: use all instances (feasible and infeasible)
    varFeatures = read_csv(
        trainfolder + "/VarFeatures_all.csv",
        header=None
    ).values[:n_Vars_small * n_Samples, :]

    conFeatures = read_csv(
        trainfolder + "/ConFeatures_all.csv",
        header=None
    ).values[:n_Cons_small * n_Samples, :]

    edgFeatures = read_csv(
        trainfolder + "/EdgeFeatures_all.csv",
        header=None
    ).values[:n_Eles_small * n_Samples, :]

    edgIndices = read_csv(
        trainfolder + "/EdgeIndices_all.csv",
        header=None
    ).values[:n_Eles_small * n_Samples, :]

    labels = read_csv(
        trainfolder + "/Labels_feas.csv",
        header=None
    ).values[:n_Samples, :]


if args.type == "obj":
    # For objective prediction: use only feasible instances
    varFeatures = read_csv(
        trainfolder + "/VarFeatures_feas.csv",
        header=None
    ).values[:n_Vars_small * n_Samples, :]

    conFeatures = read_csv(
        trainfolder + "/ConFeatures_feas.csv",
        header=None
    ).values[:n_Cons_small * n_Samples, :]

    edgFeatures = read_csv(
        trainfolder + "/EdgeFeatures_feas.csv",
        header=None
    ).values[:n_Eles_small * n_Samples, :]

    edgIndices = read_csv(
        trainfolder + "/EdgeIndices_feas.csv",
        header=None
    ).values[:n_Eles_small * n_Samples, :]

    labels = read_csv(
        trainfolder + "/Labels_obj.csv",
        header=None
    ).values[:n_Samples, :]

# Get feature dimensions
nConsF = conFeatures.shape[1]
nVarF = varFeatures.shape[1]
nEdgeF = edgFeatures.shape[1]
n_Cons = conFeatures.shape[0]
n_Vars = varFeatures.shape[0]

# Set-up device
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

# Load dataset into gpu
varFeatures = torch.tensor(varFeatures, dtype=torch.float32).to(device)
conFeatures = torch.tensor(conFeatures, dtype=torch.float32).to(device)
edgFeatures = torch.tensor(edgFeatures, dtype=torch.float32).to(device)
edgIndices = torch.tensor(edgIndices, dtype=torch.long).to(device)
edgIndices = torch.transpose(edgIndices, 0, 1).to(device)
labels = torch.tensor(labels, dtype=torch.float32).to(device)


def create_batch_data(conFeatures, edgIndices, edgFeatures, varFeatures, labels, 
                     n_samples, n_vars_small, n_cons_small, n_eles_small, batch_size=1000):
    batch_list = []
    
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples) 
        
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
            
            # Adjust edge indices to be relative to single graph (0 to n_cons_small-1, 0 to n_vars_small-1)
            sample_edge_indices = edgIndices[:, edge_start:edge_end].clone()
            sample_edge_indices[0] -= i * n_cons_small
            sample_edge_indices[1] -= i * n_vars_small
            batch_edge_indices.append(sample_edge_indices)
            
            # Collect labels
            if args.type != 'sol':
                batch_labels.append(labels[i])
            else:
                batch_labels.append(labels[var_start:var_end])
        
        # Convert lists to tensors, maintaining batch dimension
        batch_cons_tensor = torch.stack(batch_cons, dim=0)
        batch_vars_tensor = torch.stack(batch_vars, dim=0)
        batch_edges_tensor = torch.stack(batch_edges, dim=0)
        batch_edge_indices_tensor = torch.stack(batch_edge_indices, dim=0)
        
        if args.type != 'sol':
            batch_labels_tensor = torch.stack(batch_labels, dim=0)
        else:
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


# Verify actual data availability
actual_samples = labels.shape[0]
if actual_samples < n_Samples:
    n_Samples = actual_samples

# Calculate maximum available samples from each data component
max_samples_from_con = conFeatures.shape[0] // n_Cons_small
max_samples_from_var = varFeatures.shape[0] // n_Vars_small  
max_samples_from_edge = edgFeatures.shape[0] // n_Eles_small


max_samples_from_labels = labels.shape[0]

# Use minimum to ensure data consistency
actual_available_samples = min(max_samples_from_con, max_samples_from_var, 
                              max_samples_from_edge, max_samples_from_labels)
n_Samples =  min(n_Samples, actual_available_samples)


# Create batched data
batch_data_list = create_batch_data(conFeatures, edgIndices, edgFeatures, varFeatures, labels,
                                   n_Samples, n_Vars_small, n_Cons_small, n_Eles_small)

# Set Dual-Attention-Backbone hyperparameters
d_model = embSize
d_ff = embSize * 2
d_k = d_v = embSize
n_layers = 4
n_heads = 2

# Initialize model
model = Dual_Attention(embSize, nConsF, nEdgeF, nVarF,
                        n_layers, n_heads, d_model, d_k, d_v, d_ff,
                        isGraphLevel=True)


model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Calculate initial loss as baseline
loss_init, _, _ = process(model, batch_data_list, optimizer, type=args.type)
epoch = 0
count_restart = 0
err_best = 2
loss_best = 1e10

# Main training loop
while epoch <= max_epochs:
    # Train model and get metrics
    train_loss, errs, err_rate = process(
        model,
        batch_data_list,
        optimizer,
        type=args.type
    )

    if args.type == "fea":
        # For feasibility prediction, track error rate
        print(
            f"EPOCH: {epoch}, TRAIN LOSS: {train_loss}",
            f"ERRS: {errs}, ERRATE: {err_rate}"
        )
        if err_rate < err_best:
            torch.save(model.state_dict(), model_path)
            print("model saved to:", model_path)
            err_best = err_rate
    else:
        # For objective value prediction, track loss
        print(f"EPOCH: {epoch}, TRAIN LOSS: {train_loss}")
        if train_loss < loss_best:
            torch.save(model.state_dict(), model_path)
            print("model saved to:", model_path)
            loss_best = train_loss
            early_stop_counter = 0 
        else:
            early_stop_counter += 1

    epoch += 1
