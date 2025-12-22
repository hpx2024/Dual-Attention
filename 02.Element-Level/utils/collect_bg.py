import pyscipopt as scp
import pyscipopt 
import torch
import argparse
import os
import pickle
import numpy as np
import json
import gurobipy as gp

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_a_new2_gurobi(ins_name):

    # Read model using Gurobi
    gp.setParam('LogToConsole', 0)
    m = gp.read(ins_name)

    ncons = m.NumConstrs
    nvars = m.NumVars

    # Get and sort variables by name for consistent ordering
    mvars = m.getVars()
    mvars.sort(key=lambda v: v.varName)

    # Initialize variable node features
    # Features: [obj_coeff, avg_coeff_in_constraints, degree, max_coeff, min_coeff, is_binary]
    v_nodes = []
    b_vars = []  # Binary variable indices
    ori_start = 6  # Number of base features
    emb_num = 15

    for i in range(len(mvars)):
        tp = [0] * ori_start
        tp[3] = 0      # max_coeff
        tp[4] = 1e+20  # min_coeff

        # Mark binary variables
        if mvars[i].vtype == gp.GRB.BINARY:
            tp[ori_start - 1] = 1
            b_vars.append(i)
        
        v_nodes.append(tp)

    # Create variable name to index mapping
    v_map = {}
    for indx, v in enumerate(mvars):
        v_map[v.varName] = indx

    # 1.Process objective function
    obj_expr = m.getObjective()
    obj_cons = [0] * (nvars + 2)
    indices_spr = [[], []]
    values_spr = []
    obj_node = [0, 0, 0, 0] 

    # Extract objective coefficients and build sparse matrix for objective
    for i in range(obj_expr.size()):
        var = obj_expr.getVar(i)
        vnm = var.varName
        coeff = obj_expr.getCoeff(i)
        v_indx = v_map[vnm]

        obj_cons[v_indx] = coeff

        # Add edge for objective function (constraint index 0)
        if coeff != 0:
            indices_spr[0].append(0)
            indices_spr[1].append(v_indx)
            values_spr.append(1)
        
        # Update variable features with objective coefficient
        v_nodes[v_indx][0] = coeff
        obj_node[0] += coeff
        obj_node[1] += 1

    # Compute average objective coefficient (avoid division by zero)
    if obj_node[1] > 0:
        obj_node[0] /= obj_node[1]
    else:
        obj_node[0] = 0

    # 2.Process constraints
    cons = m.getConstrs()
    new_cons = []
    
    # Filter out empty constraints
    for cind, c in enumerate(cons):
        row = m.getRow(c)
        if row.size() == 0:
            continue
        new_cons.append(c)
    
    cons = new_cons
    ncons = len(cons)
    
    # Sort constraints by size (number of variables) and name for consistency
    cons_map = []
    for i, x in enumerate(cons):
        row = m.getRow(x)
        cons_name = getattr(x, 'constrName', 'unnamed')
        cons_map.append([x, row.size(), cons_name])
    
    cons_map = sorted(cons_map, key=lambda x: [x[1], x[2]])

    cons = [x[0] for x in cons_map]

    lcons = ncons
    c_nodes = []

    # Extract constraint features
    for cind, c in enumerate(cons):
        row = m.getRow(c)

        # Determine constraint sense and bounds
        if c.sense == gp.GRB.EQUAL:
            lhs = rhs = c.rhs
        elif c.sense == gp.GRB.LESS_EQUAL:
            lhs = -gp.GRB.INFINITY
            rhs = c.rhs
        elif c.sense == gp.GRB.GREATER_EQUAL:
            lhs = c.rhs
            rhs = gp.GRB.INFINITY
        else:
            lhs = -gp.GRB.INFINITY
            rhs = c.rhs

        # Encode constraint sense: 0 (<=), 1 (>=), 2 (==)
        sense = 0
        if rhs == lhs:
            sense = 2
        elif rhs >= 1e+20:
            sense = 1
            rhs = lhs

        # Process constraint coefficients
        summation = 0
        for j in range(row.size()):
            var = row.getVar(j)
            coeff_val = row.getCoeff(j)
            var_name = var.varName
            v_indx = v_map[var_name]
            # Add edge to sparse adjacency matrix
            if coeff_val != 0:
                indices_spr[0].append(cind)
                indices_spr[1].append(v_indx)
                values_spr.append(1)

            # Update variable features based on constraint coefficients
            v_nodes[v_indx][2] += 1
            v_nodes[v_indx][1] += coeff_val / lcons
            v_nodes[v_indx][3] = max(v_nodes[v_indx][3], coeff_val)
            v_nodes[v_indx][4] = min(v_nodes[v_indx][4], coeff_val)
            summation += coeff_val

        # Constraint node features: [avg_coeff, degree, rhs, sense]
        llc = max(row.size(), 1)
        c_nodes.append([summation / llc, llc, rhs, sense])

    # Add objective function as a constraint node
    c_nodes.append(obj_node)

    v_nodes = torch.as_tensor(v_nodes, dtype=torch.float32).to(device)
    c_nodes = torch.as_tensor(c_nodes, dtype=torch.float32).to(device)
    b_vars = torch.as_tensor(b_vars, dtype=torch.int32).to(device)

    # Create sparse adjacency matrix (num_constraints+1 includes objective)
    A = torch.sparse_coo_tensor(
        indices_spr, values_spr, (ncons + 1, nvars)
    ).to(device)

    # Clip outliers in variable features
    clip_max = [20000, 1, torch.max(v_nodes, 0)[0][2].item()]
    clip_min = [0, -1, 0]

    v_nodes[:, 0] = torch.clamp(v_nodes[:, 0], clip_min[0], clip_max[0])

    # Normalize variable node features to [1e-5, 1]
    maxs = torch.max(v_nodes, 0)[0]
    mins = torch.min(v_nodes, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    v_nodes = v_nodes - mins
    v_nodes = v_nodes / diff
    v_nodes = torch.clamp(v_nodes, 1e-5, 1)

    # Avoid division by zero
    maxs = torch.max(c_nodes, 0)[0]
    mins = torch.min(c_nodes, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    c_nodes = c_nodes - mins
    c_nodes = c_nodes / diff
    c_nodes = torch.clamp(c_nodes, 1e-5, 1)

    return A, v_map, v_nodes, c_nodes, b_vars


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataDir', type=str, default='./datasets/IP/train', help='The directory of the instance')
    args = parser.parse_args()

    dataDir = args.dataDir

    INS_DIR = os.path.join(dataDir,'ins')
    BG_DIR = os.path.join(dataDir,'bg')

    os.makedirs(BG_DIR, exist_ok=True)

    filenames = os.listdir(INS_DIR)

    # Process each instance file
    for filename in filenames:

        # Skip if bipartite graph already extracted
        if not os.path.exists(os.path.join(BG_DIR, filename + '.bg')):
            print(filename)
            ins_name = os.path.join(INS_DIR, filename)

            # Extract bipartite graph features
            A, v_map, v_nodes, c_nodes, b_vars = get_a_new2_gurobi(ins_name)

            # Store bipartite graph data as list
            BG_data = [A, v_map, v_nodes, c_nodes, b_vars]

            with open(os.path.join(BG_DIR, filename + '.bg'), 'wb') as f:
                pickle.dump(BG_data, f)
            print(f"Bipartite graph extraction complete for {filename}")
        
    print("\n=== All files processed successfully! ===")