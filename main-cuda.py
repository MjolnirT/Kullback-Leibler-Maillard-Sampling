import torch
import numpy as np
import sys
import time
import pickle
from utility import *
from utility_io import get_filename, read_algorithms
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

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

if __name__ == '__main__':
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_flag = True
    filename = sys.argv[1]
    environment, algorithms = read_algorithms(filename, print_flag=True)

    simulations_per_round = algorithms["BernoulliTS"]["params"]["simulation_rounds"]
    # env_reward = environment['reward']
    test_case = environment['test case']
    T_timespan = environment["base"]["T_timespan"]

    algorithms_name = list(algorithms.keys())
    n_simulations = environment['n_simulations']
    batch_size = 100
    num_batch = math.ceil(n_simulations / batch_size)
    results = simulate_batch_simulation(batch_size, num_batch, environment, algorithms, algorithms_name, device)

    print(simulate_parallel_computation(a, b))

    # filename = get_filename(T_timespan, n_simulations, test_case,
    #                         simulations_per_round, is_simulation=True)
    # filename = 'data/' + filename
    # with open(filename, 'rb') as file:
    #     records = pickle.load(file)
    # file.close()
    for record in results:
        record = record.cpu().numpy()

    filename = get_filename(T_timespan, n_simulations, test_case,
                            simulations_per_round, is_simulation=True)
    filename = 'data/' + filename
    with open(filename, 'wb') as file:
        pickle.dump(results, file)
    file.close()
    # load the simulation results
