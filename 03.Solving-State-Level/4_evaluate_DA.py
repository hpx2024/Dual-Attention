import os
import sys
import importlib
import argparse
import csv
import numpy as np
import time
import pickle

import ecole
import pyscipopt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'ufacilities', 'indset', 'mknapsack'],
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    result_file = f"{args.problem}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    instances = []
    seeds = [0, 1, 2, 3, 4]

    # SCIP internal branching heuristics to evaluate
    internal_branchers = ['relpscost']

    # DA model to evaluate
    gcnn_models = ['il']

    time_limit = 3600

    if args.problem == 'setcover':
        instances += [{'type': 'test', 'path': f"data/instances/setcover/test_400r_750c_0.05d/instance_{i+1}.lp"} for i in range(40)]
        instances += [{'type': 'transfer', 'path': f"data/instances/setcover/transfer_500r_1000c_0.05d/instance_{i+1}.lp"} for i in range(40)]

    elif args.problem == 'cauctions':
        instances += [{'type': 'test', 'path': f"data/instances/cauctions/test_100_500/instance_{i+1}.lp"} for i in range(40)]
        instances += [{'type': 'transfer', 'path': f"data/instances/cauctions/transfer_200_1000/instance_{i+1}.lp"} for i in range(40)]

    elif args.problem == 'ufacilities':
        instances += [{'type': 'test', 'path': f"data/instances/ufacilities/test_35_35_5/instance_{i+1}.lp"} for i in range(40)]
        instances += [{'type': 'transfer', 'path': f"data/instances/ufacilities/transfer_60_35_5/instance_{i+1}.lp"} for i in range(40)]

    elif args.problem == 'indset':
        instances += [{'type': 'test', 'path': f"data/instances/indset/test_500_4/instance_{i+1}.lp"} for i in range(40)]
        instances += [{'type': 'transfer', 'path': f"data/instances/indset/transfer_1000_4/instance_{i+1}.lp"} for i in range(40)]

    elif args.problem == 'mknapsack':
        instances += [{'type': 'test', 'path': f"data/instances/mknapsack/test_100_6/instance_{i+1}.lp"} for i in range(40)]
        instances += [{'type': 'transfer', 'path': f"data/instances/mknapsack/transfer_100_12/instance_{i+1}.lp"} for i in range(40)]

    else:
        raise NotImplementedError

    branching_policies = []


    for brancher in internal_branchers:
        for seed in seeds:
            branching_policies.append({
                    'type': 'internal',
                    'name': brancher,
                    'seed': seed,
             })

    for model in gcnn_models:
        for seed in seeds:
            branching_policies.append({
                'type': 'dual_attention',
                'name': model,
                'seed': seed,
            })

    print(f"problem: {args.problem}")
    print(f"gpu: {args.gpu}")
    print(f"time limit: {time_limit} s")


    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:0"

    import torch
    from actor.actor import Dual_Attention

    embSize = 64
    nConsF = 5
    nEdgeF = 1
    nVarF = 19
    d_model = 64
    d_ff = 128
    d_k = d_v = 64
    n_layers = 4
    n_heads = 2

    loaded_models = {}
    loaded_calls = {}
    for policy in branching_policies:
        if policy['type'] == 'dual_attention':
            if policy['name'] not in loaded_models:

                # Load trained weights based on training method
                model = Dual_Attention(embSize, nConsF, nEdgeF, nVarF, n_layers, n_heads, d_model, d_k, d_v, d_ff).to(device)
                if policy['name'] == 'il':
                    model.load_state_dict(torch.load(f'DA/{args.problem}/0/il.pkl'))
                else:
                    raise Exception(f"Unrecognized policy {policy['name']}")
                model.eval()
                loaded_models[policy['name']] = model
            
            # Assign cached model to policy
            policy['model'] = loaded_models[policy['name']]

    print("running SCIP...")

    # CSV output configuration
    fieldnames = [
        'policy',
        'seed',
        'type',
        'instance',
        'nnodes',
        'nlps',
        'stime',
        'gap',
        'status',
        'walltime',
        'proctime',
    ]
    os.makedirs('results_GT', exist_ok=True)
    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0,
                       'limits/time': time_limit, 'timing/clocktype': 1}

    with open(f"results_GT/{result_file}", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for instance in instances:
            print(f"{instance['type']}: {instance['path']}...")

            for policy in branching_policies:
                if policy['type'] == 'internal':
                    # Run SCIP's default brancher
                    env = ecole.environment.Configuring(scip_params={**scip_parameters,
                                                        f"branching/{policy['name']}/priority": 9999999})
                    env.seed(policy['seed'])

                    walltime = time.perf_counter()
                    proctime = time.process_time()

                    env.reset(instance['path'])
                    _, _, _, _, _ = env.step({})

                    walltime = time.perf_counter() - walltime
                    proctime = time.process_time() - proctime

                elif policy['type'] == 'dual_attention':
            
                    env = ecole.environment.Branching(observation_function=ecole.observation.NodeBipartite(),
                                                      scip_params=scip_parameters)
                    env.seed(policy['seed'])
                    torch.manual_seed(policy['seed'])

                    walltime = time.perf_counter()
                    proctime = time.process_time()

                    observation, action_set, _, done, _ = env.reset(instance['path'])
                    while not done:
                        with torch.no_grad():

                            constraint_features = torch.from_numpy(
                                observation.row_features.astype(np.float32)
                            ).to(device).unsqueeze(0)
                            
                            edge_index = torch.from_numpy(
                                observation.edge_features.indices.astype(np.int64)
                            ).to(device).unsqueeze(0)
                            
                            edge_attr = torch.from_numpy(
                                observation.edge_features.values.astype(np.float32)
                            ).to(device).view(-1, 1).unsqueeze(0)
                            
                            variable_features = torch.from_numpy(
                                observation.column_features.astype(np.float32)
                            ).to(device).unsqueeze(0)

                            # Forward pass through Dual_Attention
                            logits = policy['model'](
                                constraint_features, edge_index, edge_attr, variable_features
                            )
                            logits = logits.squeeze(0)  # Remove batch dimension

                            # logits = policy['model'](*observation)
                            action = action_set[logits[action_set.astype(np.int64)].argmax()]
                            observation, action_set, _, done, _ = env.step(action)

                    walltime = time.perf_counter() - walltime
                    proctime = time.process_time() - proctime

                scip_model = env.model.as_pyscipopt()
                stime = scip_model.getSolvingTime()
                nnodes = scip_model.getNNodes()
                nlps = scip_model.getNLPs()
                gap = scip_model.getGap()
                status = scip_model.getStatus()

                writer.writerow({
                    'policy': f"{policy['type']}:{policy['name']}",
                    'seed': policy['seed'],
                    'type': instance['type'],
                    'instance': instance['path'],
                    'nnodes': nnodes,
                    'nlps': nlps,
                    'stime': stime,
                    'gap': gap,
                    'status': status,
                    'walltime': walltime,
                    'proctime': proctime,
                })
                csvfile.flush()

                print(f"  {policy['type']}:{policy['name']} {policy['seed']} - {nnodes} nodes {nlps} lps {stime:.2f} ({walltime:.2f} wall {proctime:.2f} proc) s. {status}")
