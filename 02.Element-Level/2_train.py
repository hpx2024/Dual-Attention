import os
import torch
import torch_geometric
import random
import time
import datetime
from tqdm import tqdm
import sys
from torch.nn.utils.rnn import pad_sequence
import argparse
from config import *



torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Argument of this script
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='IP',choices=['IP','WA','IS','CA'])
args = parser.parse_args()

TaskName = args.dataset

# Create directory structure for saving models and logs
os.makedirs("./results/",exist_ok=True)
os.makedirs(f"./results/{TaskName}/",exist_ok=True)
os.makedirs(f"./results/{TaskName}/train_logs/",exist_ok=True)


current_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())

model_save_path = f'./results/{TaskName}/'
log_save_path = f"./results/{TaskName}/train_logs/"
log_file = open(
                f'{log_save_path}/-{current_time}_train.log', 'wb'
            ) 

# Training hyperparameters
LEARNING_RATE = 0.0001
NB_EPOCHS = 100 
BATCH_SIZE = 1
TBATCH = 8  # Gradient accumulation steps
NUM_WORKERS = 0
PRT_FREQUENCY = 1

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

info = confInfo[f"{TaskName}"]
DIR_SOL = os.path.join(info['trainDir'],'sol')
DIR_BG = os.path.join(info['trainDir'],'bg')

sample_names = os.listdir(DIR_BG)
sample_files = [(
                    os.path.join(DIR_BG, name),
                    os.path.join(DIR_SOL, name).replace('bg', 'sol')
                )for name in sample_names]

random.seed(0)
random.shuffle(sample_files)

train_files = sample_files[: int(0.80 * len(sample_files))]
valid_files = sample_files[int(0.80 * len(sample_files)):]

if TaskName == "IP":
    from model.model_DA import Dual_Attention
    from model.model_DA import DualDataset_addPosition as DualDataset
else:
    from model.model_DA import Dual_Attention, DualDataset

def collate_fn(batch):
    return batch

# Create data loaders
train_data = DualDataset(train_files)
train_loader = torch_geometric.loader.DataLoader(
                    train_data, batch_size=BATCH_SIZE,
                    shuffle=True, num_workers=NUM_WORKERS,
                    collate_fn=collate_fn 
                )

valid_data = DualDataset(valid_files)
valid_loader = torch_geometric.loader.DataLoader(
                    valid_data, batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=NUM_WORKERS,
                    collate_fn=collate_fn
                )

# Model hyperparameters
embSize = 64
nConsF = 4
nEdgeF = 1
if TaskName == "IP":
    nVarF = 6+12
else:
    nVarF = 6

d_model = 64
d_ff = 128
d_k = d_v = 64
n_layers = 4
n_heads = 2

PredictModel = Dual_Attention(
    embSize, nConsF, nEdgeF, nVarF, 
    n_layers,n_heads, d_model, d_k, d_v, d_ff
).to(DEVICE)

def EnergyWeightNorm(task):
    if task == "IP":
        return 1
    elif task == "WA":
        return 100
    elif task == "IS":
        return -100
    elif task == "CA":
        return -100000

def train(predict, data_loader, optimizer=None, weight_norm=1):

    # Set model mode based on whether optimizer is provided
    if optimizer:
        predict.train()  # Training mode
    else:
        predict.eval()   # Validation mode

    mean_loss = 0
    n_samples_processed = 0

    # Enable gradients only during training
    with torch.set_grad_enabled(optimizer is not None):
        for step, batch in enumerate(data_loader):
            batch = batch.to(DEVICE)

            # Extract target solutions and objective values
            solInd = batch.nsols
            target_sols = []
            target_vals = []

            # Collect solutions and objective values for each instance in batch
            for i in range(solInd.shape[0]):
                sols = batch.solutions[i]
                vals = batch.objVals[i]
                target_sols.append(sols)
                target_vals.append(vals)

            # Handle infinite values in constraint features
            mask = torch.isinf(batch.constraint_features)
            batch.constraint_features[mask] = 10

            # Prepare batched inputs for model
            constraint_features = []
            edge_index = []
            edge_attr = []
            variable_features = []
            for i in range(len(batch)):
                constraint_features.append(batch[i].constraint_features)
                edge_index.append(batch[i].edge_index)
                edge_attr.append(batch[i].edge_attr)
                variable_features.append(batch[i].variable_features)

            # Stack into batch tensors
            constraint_features = torch.stack(constraint_features, dim=0)
            edge_index = torch.stack(edge_index, dim=0)
            edge_attr = torch.stack(edge_attr, dim=0)
            variable_features = torch.stack(variable_features, dim=0)

            # Forward pass: predict binary distribution
            BD = predict(
                constraint_features.to(DEVICE),
                edge_index.to(DEVICE),
                edge_attr.to(DEVICE),
                variable_features.to(DEVICE),
            )
            BD = BD.sigmoid()  # Apply sigmoid to get probabilities

            # Compute loss
            loss = 0

            # Calculate weighted cross-entropy loss for each solution
            for ind, (sols, vals) in enumerate(zip(target_sols, target_vals)):

                # Compute importance weights based on objective values
                n_vals = vals
                exp_weight = torch.exp(-n_vals/weight_norm)
                weight = exp_weight/exp_weight.sum()

                # Get binary variable mask
                varInds = batch.varInds[ind]
                varname_map = varInds[0][0]
                b_vars = varInds[1][0].long()

                # Extract solutions for binary variables only
                sols = sols[:, varname_map][:, b_vars]

                # Get predicted probabilities for binary variables
                n_var = batch.ntvars[ind]
                pre_sols = BD[ind].squeeze()[b_vars]

                # Cross-entropy loss for variable = 1
                pos_loss = -(
                    (pre_sols + 1e-8).log()[None, :] *
                    (sols == 1).float()
                )

                # Cross-entropy loss for variable = 0
                neg_loss = -(
                    (1-pre_sols + 1e-8).log()[None, :] *
                    (sols == 0).float()
                ) 
                sum_loss = pos_loss + neg_loss

                # Weight by solution quality and sum over all variables
                sample_loss = sum_loss*weight[:, None]
                loss += sample_loss.sum()

            if optimizer is not None:
                loss.backward()

            # Update parameters every TBATCH steps (gradient accumulation)
            if step % TBATCH == TBATCH-1 or step == len(data_loader) - 1:
                if optimizer is not None:
                    optimizer.step()
                    optimizer.zero_grad()

                # Print progress
                if step % PRT_FREQUENCY == 0:
                    if str(loss.item()) == "nan":
                        print("Loss is NaN, skipping this step.")
                    mod = 'train' if optimizer else 'valid'
                    print('Epoch {} {} [{}/{}] loss {:.6f}'.format( epoch, mod, step,len(data_loader),loss.item()))

            mean_loss += loss.item()
            n_samples_processed += batch.num_graphs
    mean_loss /= n_samples_processed

    return mean_loss

optimizer = torch.optim.Adam(PredictModel.parameters(), lr=LEARNING_RATE)

weight_norm = EnergyWeightNorm(TaskName)
best_val_loss = 99999

loss_history = []

# Main training loop
for epoch in tqdm(range(NB_EPOCHS), file=sys.stdout):

    begin = time.time()
    formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Train for one epoch
    train_loss = train(PredictModel, train_loader, optimizer, weight_norm)

    # Validate
    valid_loss = train(PredictModel, valid_loader, None, weight_norm)

    tqdm.write((
        "Epoch: {0} Train loss: {1:0.3f}"
        "Valid loss: {2:0.3f}"
        ).format(epoch, train_loss, valid_loss)
    )

    # Save best model if validation loss improved
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        torch.save(
            PredictModel.state_dict(),
            model_save_path+f'model_best.pth'
        ) 

    torch.save(
        PredictModel.state_dict(),
        model_save_path+f'model_last.pth'
    )

    loss_history.append(train_loss)

    st = (
        f'Current time is:{formatted_time} , @epoch:{epoch} , '
        f'Train loss:{train_loss} , Valid loss:{valid_loss} , '
        f'TIME:{time.time()-begin}\n'
    )

    log_file.write(st.encode())
    log_file.flush() 

print('Training done!')