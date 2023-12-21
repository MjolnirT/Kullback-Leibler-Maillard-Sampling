import sys
from generate_plots import generate_plots
from utility import *
from simulation import simulate_single_simulation
from multiprocessing import Pool, cpu_count, Manager
import pickle
import time

from utility_io import get_filename, read_algorithms

if __name__ == '__main__':
    start_time = time.time()

    print_flag = True
    filename = sys.argv[1]
    environment, algorithms = read_algorithms(filename, print_flag=True)
    simulations_per_round = algorithms["BernoulliTS"]["params"]["simulation_rounds"]
    n_simulations = environment['n_simulations']
    env_reward = environment['reward']
    test_case = environment['test case']
    T_timespan = environment["base"]["T_timespan"]

    algorithms_name = list(algorithms.keys())

    # parallel simulation process
    # Use a maximum of 20 processes or the available CPU threads, whichever is smaller
    message('--- Start parallel simulation process ---', print_flag=print_flag)
    num_processes = min(16, cpu_count())
    message(f'Using CPUs: {num_processes}', print_flag=print_flag)
    pool = Pool(processes=num_processes)

    # Create a shared counter and a lock.
    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()

    # Start the pool with the modified function.
    results = pool.starmap(simulate_single_simulation,
                           [(i, counter, lock, algorithms, algorithms_name, n_simulations, env_reward) for i
                            in range(n_simulations)])

    print(f"All {n_simulations} simulations completed.")

    pool.close()
    pool.join()

    filename = get_filename(T_timespan, n_simulations, test_case,
                            simulations_per_round, is_simulation=True)
    filename = 'data/' + filename
    with open(filename, 'wb') as file:
        pickle.dump(results, file)
    file.close()

    # print out execution time
    message(f'Total time elapsed {time.time() - start_time}', print_flag)
    exclude_alg = ['KL-MS+JefferysPrior', 'MS', 'MS+']
    generate_plots(filename, env_reward, algorithms_name,
                   ref_alg='BernoulliTS',
                   exclude_alg=None)
    message(f'time elapsed {time.time() - start_time}', print_flag)
    message(f'filename: {filename}', print_flag)
