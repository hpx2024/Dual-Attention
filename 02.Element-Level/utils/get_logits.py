import os
import sys
import pickle

import torch

import time
import argparse
from collect_bg import get_a_new2_gurobi
import numpy as np


# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='IP')
args = parser.parse_args()

TaskName = args.dataset

# Set up directories
EXP_NAME = os.path.join("results",TaskName,"eval_logs")
TEST_INS = os.path.join(f"./datasets/{TaskName}/test",'ins')

if TaskName == "IP":
    from model.model_DA import Dual_Attention, postion_get
else:
    from model.model_DA import Dual_Attention

INS_DIR = TEST_INS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up output directory
now = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
exp_dir = f'logits'
output_dir = os.path.join(EXP_NAME, exp_dir)
os.makedirs(output_dir, exist_ok=True)

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

# Load trained model parameters
policy = Dual_Attention(
    embSize, nConsF, nEdgeF, nVarF, 
    n_layers,n_heads, d_model, d_k, d_v, d_ff
).to(DEVICE)
states = torch.load(os.path.join(f"results/{TaskName}/",f'model_best.pth'))
policy.load_state_dict(states)


insnames = os.listdir(f'./datasets/{TaskName}/test/ins')
file_pathList = [os.path.join(f'./datasets/{TaskName}/test/ins',insname) for insname in insnames]

for step, filpath in enumerate(file_pathList):

    # 1.Extract bipartite graph features from instance
    A, v_map, v_nodes, c_nodes, b_vars = get_a_new2_gurobi(filpath)
    constraint_features = c_nodes.cpu()

    # 2.Prepare constraint features (handle NaN values)
    if isinstance(constraint_features, torch.Tensor):
        constraint_features[torch.isnan(constraint_features)] = 1
    else:
        constraint_features[np.isnan(constraint_features)] = 1

    # 3.Prepare variable features (add position encoding if needed)
    variable_features = v_nodes
    if TaskName == "IP":
        variable_features = postion_get(variable_features)
    
    # 4.Prepare edge features
    edge_indices = A._indices()
    edge_features = A._values().unsqueeze(1)
    edge_features = torch.ones(edge_features.shape)

    constraint_features = constraint_features.unsqueeze(0)
    edge_indices = edge_indices.unsqueeze(0)
    edge_features = edge_features.unsqueeze(0)
    variable_features = variable_features.unsqueeze(0)

    # 5.Run model inference
    try:
        with torch.no_grad():
            output = policy(
                constraint_features.to(DEVICE),
                edge_indices.to(DEVICE),
                edge_features.to(DEVICE),
                variable_features.to(DEVICE),
            ).sigmoid().cpu().squeeze()
    except Exception as e:
        print(f"Error processing file {filpath}: {str(e)}")
        continue

    # Save predictions as pickle file
    pickle.dump({
        'pre':output.cpu().numpy(),
    },open(os.path.join(output_dir,f'{insnames[step]}.prob'),'wb'))


print('done')

