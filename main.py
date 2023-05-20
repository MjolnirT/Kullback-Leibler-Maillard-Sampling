from SearchOptConfig import SearchOptConfig
from generate_plots import generate_plots
from utility import *
from BernoulliTS import BernoulliTS
from BernoulliKLMS import KLMS, KLMSJefferysPrior
from MS import MS, MSPlus
from simulation import simulate_single_simulation
from multiprocessing import Pool, cpu_count, Manager
import pickle
import time

if __name__ == '__main__':
    start_time = time.time()
    print_flag = True
    env_reward = np.linspace(0.1, 0.9, 9)

    # env_reward = [0.8] + [0.9]

    # to pick the best configuration for MS+, doing a grid search from 100 simulations
    opt_config = SearchOptConfig(env_reward, n_arms=len(env_reward), n_rounds=100)

    message(f'reward_probabilities: {env_reward}', print_flag=True)

    # set algorithms and their parameters
    variance = float(1 / 4)
    T_timespan = 1000
    n_arms = env_reward.shape[0]
    n_simulations = 200

    algorithms = {'BernoulliTS':
                      {'model': BernoulliTS,
                       'params': {"n_arms": n_arms, "n_rounds": T_timespan}},
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
                                  "variance": variance, "B": opt_config[0], "C": opt_config[1], "D": opt_config[2]}}}
    algorithms_name = list(algorithms.keys())

    # parallel simulation process
    # Use a maximum of 24 processes or the available CPU threads, whichever is smaller
    num_processes = min(8, cpu_count())
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

    # results = pool.map(simulate_single_simulation, range(n_simulations))
    pool.close()
    pool.join()

    filename = 'simulation.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(results, file)

    # print out execution time
    message(f'time elapsed {time.time()-start_time}', print_flag)
    generate_plots(filename, env_reward, algorithms_name, ref_alg='BernoulliTS')
    message(f'time elapsed {time.time()-start_time}', print_flag)
