import os
import sys
import pickle
import time
from multiprocessing import Pool, cpu_count, Manager
from simulation import simulate_single_simulation
from utility_functions.generate_plots import generate_plots
from utility_functions.utility import message
from utility_functions.utility_io import check_folder_exist, get_filename, read_algorithms
from global_config import DATA_DIR, PLOT_DIR, LOG_FLAG


def main():
    start_time = time.time()
    check_folder_exist(DATA_DIR)
    check_folder_exist(PLOT_DIR)
    # Read the config file
    config_filepath = sys.argv[1]
    environment, algorithms = read_algorithms(config_filepath, print_flag=LOG_FLAG)
    simulations_per_round = algorithms["BernoulliTS"]["params"]["simulation_rounds"]
    n_simulations = environment['n_simulations']
    env_reward = environment['reward']
    test_case = environment['test case']
    T_timespan = environment["base"]["T_timespan"]

    algorithms_name = list(algorithms.keys())

    # Parallel simulation process
    message('--- Start parallel simulation process ---', print_flag=LOG_FLAG)
    num_processes = min(16, cpu_count())
    message(f'Using CPUs: {num_processes}', print_flag=LOG_FLAG)
    
    with Pool(processes=num_processes) as pool:
        # Create a shared counter and a lock
        manager = Manager()
        counter = manager.Value('i', 0)
        lock = manager.Lock()

        # Start the pool with the modified function
        results = pool.starmap(simulate_single_simulation,
                               [(i, counter, lock, algorithms, algorithms_name, n_simulations, env_reward) 
                                for i in range(n_simulations)])
    print(f"All {n_simulations} simulations completed.")

    # Save the results
    simulation_filepath = os.path.join(DATA_DIR, get_filename(T_timespan, n_simulations, test_case,
                                                              simulations_per_round, is_simulation=True))
    with open(simulation_filepath, 'wb') as file:
        pickle.dump(results, file)

    # Print out execution time
    elapsed_time = time.time() - start_time
    message(f'Total time elapsed {elapsed_time}', LOG_FLAG)
    generate_plots(simulation_filepath,
                   env_reward,
                   algorithms_name,
                   PLOT_DIR,
                   ref_alg='BernoulliTS',
                   exclude_alg=None)
    message(f'time elapsed {elapsed_time}', LOG_FLAG)
    message(f'Simulation file path: {simulation_filepath}', LOG_FLAG)

if __name__ == '__main__':
    main()
