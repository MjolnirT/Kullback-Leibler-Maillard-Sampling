import os.path
import sys
import pickle
import time
from simulation import simulate_single_simulation
from multiprocessing import Pool, cpu_count, Manager
from utility_functions.generate_plots import generate_plots
from utility_functions.utility import *
from utility_functions.utility_io import check_folder_exist
from utility_functions.utility_io import get_filename, read_algorithms

output_dir = "./data"
plot_dir = "./figures"

if __name__ == '__main__':
    start_time = time.time()

    print_flag = True
    config_filepath = sys.argv[1]
    environment, algorithms = read_algorithms(config_filepath, print_flag=True)
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

    simulation_filename = get_filename(T_timespan, n_simulations, test_case,
                                       simulations_per_round, is_simulation=True)
    simulation_filename = os.path.join(output_dir, simulation_filename)
    check_folder_exist(output_dir)
    with open(simulation_filename, 'wb') as file:
        pickle.dump(results, file)
    file.close()

    # print out execution time
    message(f'Total time elapsed {time.time() - start_time}', print_flag)
    exclude_alg = ['KL-MS+JefferysPrior', 'MS', 'MS+']
    check_folder_exist(plot_dir)
    generate_plots(simulation_filename,
                   env_reward,
                   algorithms_name,
                   plot_dir,
                   ref_alg='BernoulliTS',
                   exclude_alg=None)
    message(f'time elapsed {time.time() - start_time}', print_flag)
    message(f'Simulation file path: {simulation_filename}', print_flag)
