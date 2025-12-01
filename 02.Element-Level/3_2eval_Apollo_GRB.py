import gurobipy
from gurobipy import GRB
import argparse
import random
import os
import numpy as np
import torch
from utils.collect_bg import get_a_new2_gurobi
import datetime
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json  
import pickle

# Set random seeds for reproducibility
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-p',
    '--problem',
    default="CA-reduction",
    help='MILP instance type to process.',
)
parser.add_argument(
    '-t',
    '--threads',
    default=1,
    type=int,
    help='Number of threads to use for parallel processing.',
)
args = parser.parse_args()

TaskName = args.problem
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TestNum = 100
num_threads = args.threads

PrePath = f"results/{TaskName}/eval_logs/logits"

if TaskName == "IP":
    from model.model_DA import postion_get

# Thread locks for protecting shared resources
log_lock = threading.Lock()
result_lock = threading.Lock()

def test_hyperparam(task, step):

    if task=="IP":
        if step == 0:
            return 100,20,50
        elif step == 1:
            return 40,15,20
        elif step == 2:
            return 20,15,10
        elif step == 3:
            return 5,50,30
    elif task=="IS":
        if step == 0:
            return 50,50,50
        elif step == 1:
            return 40,15,40
        elif step == 2:
            return 20,15,30
        elif step == 3:
            return 1,5,10
    elif task=="WA":
        if step == 0:
            return 70,600,5
        elif step == 1:
            return 50,500,5
        elif step == 2:
            return 40,400,5
        elif step == 3:
            return 30,0,5
    elif task=="CA":
        if step == 0:
            return 100,0,50
        elif step == 1:
            return 50,0,50
        elif step == 2:
            return 40,0,40
        elif step == 3:
            return 30,0,30

def get_graph_representation(model):

    A, v_map, v_nodes, c_nodes, b_vars = get_a_new2_gurobi(model)

    # Process constraint features
    constraint_features = c_nodes.cpu()
    constraint_features[torch.isnan(constraint_features)] = 1

    # Process variable features
    variable_features = v_nodes

    # Process edge features
    edge_indices = A._indices()
    edge_features = A._values().unsqueeze(1)
    edge_features = torch.ones(edge_features.shape)
    return A, v_map, v_nodes, c_nodes, b_vars, \
        constraint_features, variable_features, edge_indices, edge_features

def variable_alignment(v_map, b_vars, BD, fixing_status):

    all_varname = []
    for name in v_map:
        all_varname.append(name)
        fixing_status[name] = -1

    # Identify binary variables
    binary_name = [all_varname[i] for i in b_vars]

    # Create scores list
    scores = []
    for i in range(len(v_map)):
        type = "C"
        if all_varname[i] in binary_name:
            type = 'BINARY'
            fixing_status[all_varname[i]] = 2
        scores.append([i, all_varname[i], BD[i].item(), -1, type])
    
    # Sort by prediction score (descending)
    scores.sort(key=lambda x: x[2], reverse=True)

    # Keep only binary variables
    scores = [x for x in scores if x[4] == 'BINARY']
    return scores, fixing_status

def fix_variable(scores, fixing_status, k_0, k_1, delta):

    # Fix top k_1 variables at 1 (highest confidence)
    count1 = 0
    for i in range(len(scores)):
        if count1 < k_1:
            scores[i][3] = 1
            count1 += 1
            fixing_status[scores[i][1]] = 1

    # Sort by prediction score (ascending) to find low-confidence 0s
    scores.sort(key=lambda x: x[2], reverse=False)

     # Fix bottom k_0 variables at 0 (lowest confidence)
    count0 = 0
    for i in range(len(scores)):
        if count0 < k_0:
            scores[i][3] = 0
            count0 += 1
            fixing_status[scores[i][1]] = 1

    print(f'instance: {TaskName}, '
          f'fix {k_0} 0s and '
          f'fix {k_1} 1s, delta {delta}. ')
   
    return scores, fixing_status

def process_single_instance(ins_num, test_ins_name, policy, log_folder, time_limit):

    print(f"Thread {threading.current_thread().name} processing instance {ins_num}: {test_ins_name}")

    # Global tracking variables
    global_best_solution = None 
    continuous_log = []
    cumulative_time = 0
    
    ins_name_read = f'datasets/{TaskName}/test/ins/{test_ins_name}'
    os.makedirs(f'{log_folder}/{test_ins_name}', exist_ok=True)
    
    # Initialize best objective (depends on min/max problem)
    if args.problem == 'IS' or args.problem == 'CA':
        best_obj = 0
    else:
        best_obj = 1000000
    
    # Track variable fixing states
    fixing_status = {}
    
    # Storage for results
    instance_results = {
        'obj_list': [],
        'time_list': [],
        'node_list': [],
        'best_obj': best_obj
    }
    

    for step in range(len(time_limit)):

        step_start = time.time()
        
        # Extract graph features
        A, v_map, v_nodes, c_nodes, b_vars, \
            constraint_features, variable_features, edge_indices, edge_features = get_graph_representation(ins_name_read)
       
        if TaskName == "IP":
            variable_features = postion_get(variable_features)
        
        # Load pre-computed predictions
        pre_path = os.path.join(PrePath, test_ins_name) + '.prob'
        data = pickle.load(open(pre_path, 'rb'))
        BD = data['pre']

        # Get hyperparameters for this iteration
        k_0, k_1, delta = test_hyperparam(TaskName, step)

        # Align variables and determine which to fix
        scores, fixing_status = variable_alignment(v_map, b_vars, BD, fixing_status)
        del BD
        scores, fixing_status = fix_variable(scores, fixing_status, k_0, k_1, delta)

        # Read instance with Gurobi
        gurobipy.setParam('LogToConsole', 0)
        m = gurobipy.read(ins_name_read)
        m_ps = m.copy()

        # Configure solver
        m_ps.Params.TimeLimit = time_limit[step]
        m_ps.Params.Threads = 1
        m_ps.Params.MIPFocus = 1
        m_ps.Params.LogFile = f'{log_folder}/{test_ins_name}/{step}.log'

        # Create variable mapping
        instance_variabels = m_ps.getVars()
        instance_variabels.sort(key=lambda v: v.VarName)
        variabels_map = {}
        
        for v in instance_variabels:
            variabels_map[v.VarName] = v
        
        # Add trust region constraints
        alphas = []
        for i in range(len(scores)):
            tar_var = variabels_map[scores[i][1]]
            x_star = scores[i][3]

            # Skip if unfixed or already permanently fixed
            if x_star < 0 or fixing_status[scores[i][1]] != 2:
                continue
              
            tmp_var = m_ps.addVar(name=f'alp_{tar_var}', vtype=GRB.CONTINUOUS)
            alphas.append(tmp_var)
            m_ps.addConstr(tmp_var >= tar_var - x_star, name=f'alpha_up_{i}')
            m_ps.addConstr(tmp_var >= x_star - tar_var, name=f'alpha_dowm_{i}')


        # Set MIP start (warm start from previous iteration)
        if global_best_solution is not None:
            try:
                for var_name, var_value in global_best_solution.items():
                    if var_name in variabels_map:
                        variabels_map[var_name].Start = var_value 
                print(f"Set MIP start for step {step}")
            except Exception as e:
                print(f"Warning: Could not set MIP start: {e}")

        # Define callback for continuous solution logging
        def solution_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                # Time relative to this step
                step_relative_time = model.cbGet(GRB.Callback.RUNTIME)
                # Accumulated time across all steps
                accumulated_time = cumulative_time + step_relative_time
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                
                continuous_log.append({
                    'time': accumulated_time,
                    'objective': obj_val,
                    'step': step
                })
                print(f"Step {step} - Solution at {accumulated_time:.1f}s: obj={obj_val:.2f}")
        
        # Add trust region global constraint
        all_tmp = 0
        for tmp in alphas:
            all_tmp += tmp
        m_ps.addConstr(all_tmp <= delta, name="sum_alpha")

        # Solve with callback
        m_ps.optimize(solution_callback)

        step_time = time.time() - step_start

        # Update global best solution for next iteration warm start
        if m_ps.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and m_ps.SolCount > 0:
            current_obj = m_ps.ObjVal
            
            # Check if improved
            is_improved = False
            if args.problem in ['IS', 'CA']: 
                is_improved = (current_obj > instance_results['best_obj'])
            else:
                is_improved = (current_obj < instance_results['best_obj'])
            
            if is_improved or global_best_solution is None:
                # Save new best solution
                global_best_solution = {}
                for var in m_ps.getVars():
                    global_best_solution[var.VarName] = var.X
                print(f"Step {step}: New global best = {current_obj:.2f}")

        # Process variables marked for fixing at 0
        scores.sort(key=lambda x: x[2], reverse=False)
        counting = 0
        fix_count = 0
        for i in range(len(scores)):
            if scores[i][3] != 0:
                continue

            # If temporarily fixed and solution matches prediction
            if fixing_status[scores[i][1]] == 1 and scores[i][3] == 0:
                var = m.getVarByName(scores[i][1])
                if scores[i][3] == m_ps.getVarByName(scores[i][1]).x:
                    # Permanently fix this variable
                    fixing_status[scores[i][1]] = 0
                    var.lb = m_ps.getVarByName(scores[i][1]).x
                    var.ub = m_ps.getVarByName(scores[i][1]).x
                    fix_count += 1
                else:
                    # Solution didn't match, keep as binary unfixed
                    fixing_status[scores[i][1]] = 2
            counting += 1
            if counting >= k_0:
                break

        print(f'Thread {threading.current_thread().name} - {test_ins_name} step {step}: inspect {counting} vars and fix {fix_count} to be 0')

        # Process variables marked for fixing at 1
        counting = 0
        fix_count = 0
        scores.sort(key=lambda x: x[2], reverse=True)
        for i in range(len(scores)):
            if scores[i][3] != 1:
                continue
            
            if fixing_status[scores[i][1]] == 1 and scores[i][3] == 1:
                var = m.getVarByName(scores[i][1])
                if scores[i][3] == m_ps.getVarByName(scores[i][1]).x:
                    # Process variables marked for fixing at 1
                    fixing_status[scores[i][1]] = 0
                    var.lb = m_ps.getVarByName(scores[i][1]).x
                    var.ub = m_ps.getVarByName(scores[i][1]).x
                    fix_count += 1
                else:
                    # Solution didn't match
                    fixing_status[scores[i][1]] = 2
            counting += 1
            if counting >= k_1:
                break
        
        print(f'Thread {threading.current_thread().name} - {test_ins_name} step {step}: inspect {counting} vars and fix {fix_count} to be 1')

        # Save modified model for next iteration
        m.write(f'{log_folder}/{test_ins_name}/{step}.lp')
        ins_name_read = f'{log_folder}/{test_ins_name}/{step}.lp'
        
        # Update cumulative time
        cumulative_time += time_limit[step]
        print(f"Step {step} completed. Cumulative time: {cumulative_time}s")

        # Update cumulative time
        if args.problem == 'IS' or args.problem == 'IS':
            if m_ps.ObjVal > instance_results['best_obj']:
                instance_results['best_obj'] = m_ps.ObjVal
        else:
            if m_ps.ObjVal < instance_results['best_obj']:
                instance_results['best_obj'] = m_ps.ObjVal

        print(f'Thread {threading.current_thread().name} - {test_ins_name} step {step}: '
              f'obj: {m_ps.ObjVal}, time: {step_time:.2f}s, nodes: {m_ps.NodeCount}')

        # Store results
        instance_results['obj_list'].append(m_ps.ObjVal)
        instance_results['time_list'].append(step_time)
        instance_results['node_list'].append(m_ps.NodeCount)

    # Save continuous solution log
    with open(f'{log_folder}/{test_ins_name}/continuous.json', 'w') as f:
        json.dump(continuous_log, f, indent=2)

    print(f"Instance {test_ins_name}: Recorded {len(continuous_log)} continuous solutions")

    return ins_num, test_ins_name, instance_results

def prediction_correction():

    # Set up log folders
    now = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    solver = 'GRB'
    test_task = f'{TaskName}_{solver}_Apollo-{now}'

    os.makedirs("./results/",exist_ok=True)
    os.makedirs(f"./results/{TaskName}/",exist_ok=True)
    os.makedirs(f"./results/{TaskName}/eval_logs/",exist_ok=True)
    os.makedirs(f"./results/{TaskName}/eval_logs//{test_task}",exist_ok=True)
    log_folder = f'./results/{TaskName}/eval_logs/{test_task}'

    # Save hyperparameters to file
    hyperparam_file = f'{log_folder}/hyperparameters.txt'
    with open(hyperparam_file, 'w') as f:
        f.write(f"Task: {TaskName}\n")
        f.write("Hyperparameters (k_0, k_1, delta):\n")
        for step in range(4): 
            k_0, k_1, delta = test_hyperparam(TaskName, step)
            f.write(f"step {step}: {k_0},{k_1},{delta}\n")
    print(f"Hyperparameters saved to: {hyperparam_file}")

    # Model is not loaded - using pre-computed predictions
    policy = None

    log_file = open(f'{log_folder}/test.log', 'wb')

    all_obj_list = []
    all_time_list = []
    all_node_list = []
    all_best_obj_list = []
    time_limit = [100, 100, 200, 600]

    sample_names = sorted(os.listdir(f'datasets/{TaskName}/test/ins'))
    total_instances = min(len(sample_names), TestNum)
    print(f"Processing {total_instances} instances with {num_threads} threads...")

    # Use thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=num_threads) as executor:

        futures = []
        for ins_num in range(total_instances):
            test_ins_name = sample_names[ins_num]
            future = executor.submit(process_single_instance, ins_num, test_ins_name, policy, log_folder, time_limit)
            futures.append(future)

            if ins_num < total_instances - 1:
                print(f"Submitted instance {ins_num}: {test_ins_name}, waiting 15s before next submission...")
                time.sleep(15)
        
        # Collect results as threads complete
        for future in as_completed(futures):
            try:
                ins_num, test_ins_name, instance_results = future.result()
                
                with result_lock:
                    all_obj_list.extend(instance_results['obj_list'])
                    all_time_list.extend(instance_results['time_list'])
                    all_node_list.extend(instance_results['node_list'])
                    all_best_obj_list.append(instance_results['best_obj'])
                
                with log_lock:
                    for i, (obj, time_val, nodes) in enumerate(zip(
                        instance_results['obj_list'], 
                        instance_results['time_list'], 
                        instance_results['node_list']
                    )):
                        st = f'{test_ins_name}_step_{i} {obj} {time_val} {nodes}\n'
                        log_file.write(st.encode())
                        log_file.flush()
                
                print(f"Completed processing instance {ins_num}: {test_ins_name}")
                
            except Exception as e:
                print(f"Error processing instance: {e}")

    # Write summary statistics
    with log_lock:
        st = f'AVERAGE {np.mean(all_obj_list)} {np.mean(all_time_list)} {np.mean(all_node_list)} {np.mean(all_best_obj_list)}\n'
        log_file.write(st.encode())
        
        st = f'HYPERPARAMS {[test_hyperparam(TaskName, step) for step in range(len(time_limit))]}\n'
        log_file.write(st.encode())
        log_file.flush()

    log_file.close()
    print(f"All instances processed. Results saved to {log_folder}")

if __name__ == '__main__':
    prediction_correction()