import os.path
import pickle
from multiprocessing import Process, Queue
import gurobipy
import gurobipy as gp
import numpy as np
import argparse
import multiprocessing
import traceback
import time

import signal

# Prevent zombie processes
signal.signal(signal.SIGCHLD, signal.SIG_IGN)

def solve(filepath, log_dir, settings):
    try:
        # Suppress Gurobi console output
        gp.setParam('LogToConsole', 0)
        m = gurobipy.read(filepath)

        # Configure solution pool parameters
        m.Params.PoolSolutions = settings['maxsol']
        m.Params.PoolSearchMode = settings['mode']
        m.Params.TimeLimit = settings['maxtime']
        m.Params.Threads = settings['threads']

        # Set up log file
        log_path = os.path.join(log_dir, os.path.basename(filepath)+'.log')
        with open(log_path,'w'):
            pass
        m.Params.LogFile = log_path

        # Solve the model
        m.optimize()

        # Extract solutions from pool
        sols = []
        objs = []
        solc = m.getAttr('SolCount')
        oriVarNames = [var.varName for var in m.getVars()]
        
        for sn in range(solc):
            m.Params.SolutionNumber = sn
            sols.append(np.array(m.Xn))
            objs.append(m.PoolObjVal)

        # Convert to numpy arrays
        sols = np.array(sols,dtype=np.float32)
        objs = np.array(objs,dtype=np.float32)

        # Package solution data
        sol_data = {
            'var_names': oriVarNames,
            'sols': sols,
            'objs': objs,
        }

        return sol_data
    except Exception as e:
        print(f"Error solving {filepath}: {str(e)}")
        traceback.print_exc()
        return None

def collect(worker_id, ins_dir, q, sol_dir, log_dir, settings):
    processed_count = 0
    print(f"Worker {worker_id} started")
    
    while True:
        try:
            # Get next filename from queue
            filename = q.get(timeout=5)

            # None signals end of work
            if filename is None:
                print(f"Worker {worker_id} received stop signal")
                break
                
            filepath = os.path.join(ins_dir, filename)
            print(f"Worker {worker_id} processing {filename}")
            
            # Solve instance and collect solutions
            sol_data = solve(filepath, log_dir, settings)
            
            if sol_data is not None:
                # Save solution data as pickle file
                pickle.dump(sol_data, open(os.path.join(sol_dir, filename+'.sol'), 'wb'))
                processed_count += 1
                print(f'Worker {worker_id} processed {filename}, collected {len(sol_data["sols"])} solutions, queue size: {q.qsize()}')
            else:
                print(f'Worker {worker_id} failed to process {filename}')
                
        except Exception as e:
            # Timeout is expected when queue is empty
            if "timeout" not in str(e).lower():
                print(f"Worker {worker_id} error: {str(e)}")
                traceback.print_exc()
            break
    
    print(f"Worker {worker_id} finished, processed {processed_count} files")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataDir', type=str, default='./datasets/IP/train',help='The training directory of the dataset')
    parser.add_argument('--nWorkers', type=int, default=20,help='number of processes to solve distinct instances in parallel')
    parser.add_argument('--maxTime', type=int, default=3600,help='time limit of the solving process')
    parser.add_argument('--maxStoredSol', type=int, default=10,help='max number of solutions to store')
    parser.add_argument('--threads', type=int, default=1,help='number of threads used to solve a single instance')

    args = parser.parse_args()

    dataDir = args.dataDir

    # Set up directory
    INS_DIR = os.path.join(dataDir,'ins')
    SOL_DIR = os.path.join(dataDir,'sol')
    LOG_DIR = os.path.join(dataDir,'logs')

    os.makedirs(SOL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    N_WORKERS = args.nWorkers

    # Gurobi solver settings
    SETTINGS = {
        'maxtime': args.maxTime,
        'mode': 2,
        'maxsol': args.maxStoredSol,
        'threads': args.threads,
    }

    filenames = sorted(os.listdir(INS_DIR))
    print(f"Total files found: {len(filenames)}")

    # Check which files need processing (skip already processed)
    files_to_process = []
    for filename in filenames:
        if not os.path.exists(os.path.join(SOL_DIR, filename+'.sol')):
            files_to_process.append(filename)
    
    print(f"Files to process: {len(files_to_process)}")
    
    if len(files_to_process) == 0:
        print("No files need processing. All .sol files already exist.")
        exit(0)

    # Create queue and add filenames
    q = Queue()
    for filename in files_to_process:
        q.put(filename)
    
    print(f"Added {len(files_to_process)} files to queue")

    # Adjust number of workers based on workload
    actual_workers = min(N_WORKERS, len(files_to_process))
    print(f"Using {actual_workers} workers")

    # Adjust number of workers based on workload
    for i in range(actual_workers):
        q.put(None)

    # Start worker processes
    ps = []
    for i in range(actual_workers):
        p = Process(target=collect, args=(i, INS_DIR, q, SOL_DIR, LOG_DIR, SETTINGS))
        p.start()
        ps.append(p)

    # Monitor process status periodically
    # start_time = time.time()
    # while any(p.is_alive() for p in ps):
    #     time.sleep(10)
    #     alive_count = sum(1 for p in ps if p.is_alive())
    #     elapsed = time.time() - start_time
    #     print(f"Status check: {alive_count} workers alive, elapsed time: {elapsed:.1f}s")

    # Wait for all processes to complete
    for p in ps:
        p.join()

    print('All workers completed')