from SearchOptConfig import SearchOptConfig
from generate_plots import generate_plots
from utility import *
from BernoulliTS import BernoulliTS, simuBernoulliTS
from BernoulliKLMS import KLMS, KLMSJefferysPrior
from MS import MS, MSPlus
from simulation import simulate_single_simulation
from multiprocessing import Pool, cpu_count, Manager
import pickle
import time

if __name__ == '__main__':
    start_time = time.time()
    print_flag = True
    # env_reward = [0,2, 0,25]
    # test_case = 1

    # env_reward = [0.8] + [0.9]
    # test_case = 2

    env_reward = np.linspace(0.1, 0.9, 9)
    test_case = 3
    # to pick the best configuration for MS+, doing a grid search from 100 simulations
    opt_config = SearchOptConfig(env_reward, n_arms=len(env_reward), n_rounds=100)

    message(f'reward_probabilities: {env_reward}', print_flag=True)

    # set algorithms and their parameters
    variance = float(1 / 4)
    T_timespan = 10000
    n_arms = len(env_reward)
    n_simulations = 2000
    simulations_per_round = 1000
    simulation_points = 20
    algorithms = {'BernoulliTS':
                      {'model': BernoulliTS,
                       'params': {"n_arms": n_arms, "n_rounds": T_timespan, "simulation_rounds": simulations_per_round}},
                  'KL-MS':
                      {'model': KLMS,
                       'params': {"n_arms": n_arms, "n_rounds": T_timespan}},
                  'KL-MS+JefferysPrior':
                      {'model': KLMSJefferysPrior,
                       'params': {"n_arms": n_arms, "n_rounds": T_timespan}},
                  'MS':
                      {'model': MS,
                       'params': {"n_arms": n_arms, "n_rounds": T_timespan,
                                  "variance": variance}},
                  'MS+':
                      {'model': MSPlus,
                       'params': {"n_arms": n_arms, "n_rounds": T_timespan,
                                  "variance": variance, "B": opt_config[0], "C": opt_config[1], "D": opt_config[2]}},
                  'BernoulliTS+RiemannApprox':
                      {'model': simuBernoulliTS,
                       'params': {"n_arms": n_arms, "n_rounds": T_timespan, "simulation_rounds": simulation_points}}
                  }
    algorithms_name = list(algorithms.keys())

    # parallel simulation process
    # Use a maximum of 20 processes or the available CPU threads, whichever is smaller
    num_processes = min(20, cpu_count())
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

    filename = 'simulation' + '_T_' + str(T_timespan) + \
               '_s_' + str(n_simulations) + \
               '_test' + str(test_case) + \
               '_MC_' + str(simulations_per_round) + \
               '_p_' + str(simulation_points) + \
               '.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(results, file)

    # print out execution time
    message(f'time elapsed {time.time() - start_time}', print_flag)
    generate_plots(filename, env_reward, algorithms_name,
                   ref_alg='BernoulliTS',
                   exclude_alg=['BernoulliTS', 'KL-MS+JefferysPrior', 'MS', 'MS+'])
    message(f'time elapsed {time.time() - start_time}', print_flag)
    message(f'filename: {filename}', print_flag)
