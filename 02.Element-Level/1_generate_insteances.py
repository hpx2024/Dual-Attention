
import os
from ecole.instance import IndependentSetGenerator, CombinatorialAuctionGenerator
import time

def generate_ca_instances(output_dir,task_name,num):

    # Create output directory if it doesn't exist
    full_output_dir = os.path.join(output_dir, task_name)
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)

    # Initialize Combinatorial Auction generator
    generator = CombinatorialAuctionGenerator(n_items=8000, n_bids=3000, min_value=1, max_value=100)

    # Generate and save instances
    for i in range(num):
        instance = next(generator)
        instance.write_problem(os.path.join(full_output_dir, f"combinatorial-auction-{i:04}.lp"))

def generate_is_instances(output_dir,task_name,num):

    # Create output directory if it doesn't exist
    full_output_dir = os.path.join(output_dir, task_name)
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)

    # Initialize Independent Set generator
    generator = IndependentSetGenerator(n_nodes=3000, edge_probability=0.1)

    # Generate and save instances
    for i in range(num):
        instance = next(generator)
        instance.write_problem(os.path.join(full_output_dir, f"independent-set-{i:04}.lp"))

if __name__ == '__main__':
    # Record start time
    start = time.time()
    print(f"Start time: {start:.4f} seconds")

    # Generate training instances
    output_dir = "./instance/train"
    generate_ca_instances(output_dir,'CA',400)
    generate_is_instances(output_dir,'IS',400)

    # Generate testing instances
    output_dir = "./instance/test"
    generate_ca_instances(output_dir,'CA',100)
    generate_is_instances(output_dir,'IS',100)

    # Report total elapsed time
    print(f"Elapsed time: {time.time() - start:.4f} seconds")