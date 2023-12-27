import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import numpy as np
import sys
import time
import pickle
from utility import *
from utility_io import get_filename, read_algorithms, read_batch_algorithms
from simulation import simulate_batch_simulation
import math


def simulate_parallel_computation(a, b):
    # Convert inputs to PyTorch tensors and move to GPU
    a = torch.tensor(a).cuda()
    b = torch.tensor(b).cuda()

    # Perform computation on the GPU
    c = a * b  # This operation is performed in parallel on the GPU

    # Move the result back to the CPU and convert to a regular Python type, if necessary
    c = c.cpu().numpy()

    return c


if __name__ == '__main__':
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 500

    print_flag = True
    filename = sys.argv[1]
    environment, algorithms = read_batch_algorithms(filename, batch_size=batch_size, print_flag=True, device=device)

    # simulations_per_round = algorithms["BernoulliTS"]["params"]["simulation_rounds"]
    simulations_per_round = 4000

    # env_reward = environment['reward']
    test_case = environment['test case']
    T_timespan = environment["base"]["T_timespan"]

    algorithms_name = list(algorithms.keys())
    n_simulations = environment['n_simulations']
    
    num_batch = math.ceil(n_simulations / batch_size)
    results = simulate_batch_simulation(batch_size, num_batch, environment, algorithms, algorithms_name, device)


    for record in results:
        record = record.cpu().numpy()

    message(f'All {n_simulations} simulations completed.', print_flag=print_flag)
    filename = get_filename(T_timespan, n_simulations, test_case,
                            simulations_per_round, is_simulation=True)
    filename = 'data/' + filename
    with open(filename, 'wb') as file:
        pickle.dump(results, file)
    file.close()

    message(f'Total time elapsed {time.time() - start_time}', print_flag=print_flag)
    # load the simulation results
