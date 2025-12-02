import gurobipy
from gurobipy import GRB
import random
import os
import numpy as np
import torch
import argparse
from utils.collect_bg import get_a_new2_gurobi
import time
import os
from multiprocessing import Process, Queue
import multiprocessing
import pickle

# Set device and random seeds for reproducibility
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='IP',choices=['IP','WA','IS','CA'])
parser.add_argument('--nWorkers', type=int, default=5,help='number of processes to solve distinct instances in parallel')
args = parser.parse_args()

TaskName = args.dataset
TestNum = 100

N_WORKERS = args.nWorkers

INS_DIR = f"datasets/{TaskName}/test/ins"

def test_hyperparam(task):
    # set the hyperparams
    # k_0, k_1, delta
    if task == "IP":
        return 400, 5, 40
    elif task == "WA":
        return 0, 500, 40
    elif task == "IS":
        return 400, 400, 40
    elif task == "CA":
        return 60, 0, 40

k_0, k_1, delta = test_hyperparam(TaskName)

sample_names = sorted(os.listdir(f'./datasets/{TaskName}/test/ins'))
file_names = sorted(os.listdir(f'./datasets/{TaskName}/test/ins'))
file_paths = [ os.path.join(f'./datasets/{TaskName}/test/ins', filename) for filename in file_names]

# Path to pre-computed model predictions
PrePath = f"results/{TaskName}/eval_logs/logits"

def solve(log_dir, filpath, pre_path, filename):

    # Extract bipartite graph features from instance
    A, v_map, v_nodes, c_nodes, b_vars = get_a_new2_gurobi(filpath)

    # Load predictions for variable values
    data = pickle.load(open(pre_path,'rb'))
    BD = data['pre']

    # Create list of all variable names in sorted order
    all_varname = []
    for name in v_map:
        all_varname.append(name)
    
    # Get names of binary variables
    binary_name = [all_varname[i] for i in b_vars]

    # Create scores list: [index, var_name, prediction_prob, fix_value, var_type]
    scores = []
    for i in range(len(v_map)):
        type = "C"
        if all_varname[i] in binary_name:
            type = 'BINARY'
        scores.append([i, all_varname[i], BD[i].item(), -1, type])

    # Clean up to free GPU memory
    del  BD

    # Sort by prediction score (descending) and filter binary variables only
    scores.sort(key=lambda x: x[2], reverse=True)
    scores = [x for x in scores if x[4] == 'BINARY']

    # Fix k_1 variables with highest prediction scores to 1
    fixer = 0
    count1 = 0
    for i in range(len(scores)):
        if count1 < k_1:
            scores[i][3] = 1
            count1 += 1
            fixer += 1

    # Sort by prediction score (ascending) to find low-confidence 0s
    scores.sort(key=lambda x: x[2], reverse=False)

    # Sort by prediction score (ascending) to find low-confidence 0s
    count0 = 0
    for i in range(len(scores)):
        if count0 < k_0:
            scores[i][3] = 0
            count0 += 1
            fixer += 1

    print(
        f'instance: {filename}, '
        f'fix {k_0} 0s and '
        f'fix {k_1} 1s, delta {delta}. '
    )

    # Read MILP instance with Gurobi
    gurobipy.setParam('LogToConsole', 1)
    m = gurobipy.read(filpath)

    # Configure Gurobi parameters
    m.Params.TimeLimit = 1000
    m.Params.Threads = 1
    m.Params.MIPFocus = 1
    m.Params.LogFile = f'{log_dir}/{filename}.log'

    # Create variable name to Gurobi variable mapping
    instance_variabels = m.getVars()
    instance_variabels.sort(key=lambda v: v.VarName)
    variabels_map = {}
    for v in instance_variabels:
        variabels_map[v.VarName] = v
    
    # Implement trust region method
    alphas = []
    for i in range(len(scores)):
        tar_var = variabels_map[scores[i][1]]
        x_star = scores[i][3]

        if x_star < 0:
            continue
        # Add auxiliary variable alpha to measure deviation |x - x*|
        tmp_var = m.addVar(name=f'alp_{tar_var}', vtype=GRB.CONTINUOUS)
        alphas.append(tmp_var)

        # Constraints: alpha >= |x - x*|
        m.addConstr(tmp_var >= tar_var - x_star, name=f'alpha_up_{i}')
        m.addConstr(tmp_var >= x_star - tar_var, name=f'alpha_dowm_{i}')
    
    # Constraints: alpha >= |x - x*|
    all_tmp = 0
    for tmp in alphas:
        all_tmp += tmp
    m.addConstr(all_tmp <= delta, name="sum_alpha")

    # Solve the restricted problem
    m.optimize()

def collect(log_dir, ins_dir,q):

    while True:
        filename = q.get()

        # None signals end of work
        if not filename:
            break
        filepath = os.path.join(ins_dir, filename)
        pre_path = os.path.join(PrePath, filename)+'.prob'

        # Solve instance with Predict & Search
        solve(log_dir, filepath, pre_path,filename)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    # Create output directories
    solver = 'GRB'
    now = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    test_task = f'{TaskName}_{solver}_Predect&Search-k_0={k_0}-k_1={k_1}-delta={delta}-{now}'

    os.makedirs("./results/",exist_ok=True)
    os.makedirs(f"./results/{TaskName}/",exist_ok=True)
    os.makedirs(f"./results/{TaskName}/eval_logs/",exist_ok=True)
    os.makedirs(f"./results/{TaskName}/eval_logs/{test_task}",exist_ok=True)

    log_folder = f"./results/{TaskName}/eval_logs/{test_task}"

    # Create output directories (only in main process)
    q = Queue()
    for filenames in file_names:
        q.put(filenames)

    # Add stop signals
    for i in range(N_WORKERS):
        q.put(None)

    # Start worker processes with staggered start times
    ps = []
    for i in range(N_WORKERS):
        p = Process(target=collect,args=(log_folder,INS_DIR, q))
        p.start()
        ps.append(p)

        # Stagger process starts to avoid CPU congestion (except last worker)
        if i < N_WORKERS - 1:
            print(f"Started worker {i+1}/{N_WORKERS}, waiting 30s before next...")
            time.sleep(15)

    # Wait for all processes to complete
    for p in ps:
        p.join()

    print('All instances solved!')
