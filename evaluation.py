import json
import pickle
import sys
from multiprocessing import Pool, cpu_count, Manager
from generate_eval_plots import generate_eval_plots
from Base import Uniform
from utility import *
import time
import torch
from utility_io import get_filename


def evaluate_one_alg(env_reward, n_arms, n_rounds, algorithm, output_all_arm_prob=False):
    '''
    :param env_reward: numpy array with shape [n_rounds, n_arms]
    :param n_arms: scalar, the number of arms
    :param n_rounds: scalar, the number of time steps
    :param algorithm: dictionary
    :param output_all_arm_prob: boolean
    :return:
    '''
    rewards = []
    selected_arms = []
    best_reward = np.max(env_reward) * np.ones(n_rounds)
    arm_probs = np.zeros(shape=[n_rounds, n_arms])

    for t in range(n_rounds):
        chosen_arm = None
        # In the first n_arms rounds, play each arm once
        if t < n_arms:
            chosen_arm = t

        # After the first n_arms rounds, use the algorithm to select an arm
        if t >= n_arms:
            chosen_arm = algorithm.select_arm()

        # sample a return_reward based on the recorded reward
        return_reward = env_reward[t][chosen_arm]
        algorithm.update(chosen_arm, return_reward)

        # record the results
        selected_arms.append(chosen_arm)
        rewards.append(return_reward)

        # record the probability of each arm
        if output_all_arm_prob:
            arm_probs[t] = algorithm.get_arm_prob()

    arm_probs[-1] = algorithm.get_arm_prob()

    return selected_arms, rewards, best_reward, arm_probs


def evaluate_single_simulation(sim_idx, counter, lock, repeat,
                               records, eval_algorithm, eval_algorithm_name,
                               n_simulations, n_algorithms, algorithms_name, T_timespan, n_arms, env_reward):
    '''
    :param eval_algorithm: dictionary. Store the algorithm and its corresponding model and parameters needed to be evaluated
    :param eval_algorithm_name: list of string. Store the name of the algorithm
    :param env_reward: numpy array with shape [number of algorithm wait to evaluate, T_timespan, n_arms]
    '''
    selected_arm_all = np.zeros(shape=[n_algorithms, T_timespan])
    rewards_all = np.zeros(shape=[n_algorithms, T_timespan])

    # construct IPW reward
    select_arms, regrets, arm_probs, expect_rewards, time_cost = records
    # select_arms, regrets, arm_probs, expect_rewards = records
    rewards = np.max(env_reward) - regrets

    padding = 1 / n_simulations * 0.5

    ipw_reward = np.zeros(shape=[n_algorithms, T_timespan, n_arms])
    for alg_idx in range(n_algorithms):
        for t in range(n_arms, T_timespan):
            chosen_arm = select_arms[alg_idx, t].astype(int)
            arm_prob = arm_probs[alg_idx, t, chosen_arm]
            if 'BernoulliTS' in algorithms_name[alg_idx]:
                if arm_probs[alg_idx, t, chosen_arm] < padding:
                    arm_prob = padding

            ipw_reward[alg_idx, t, chosen_arm] = \
                rewards[alg_idx, t] / arm_prob

    for alg_idx in range(n_algorithms):
        # model = algorithm(*args)
        model = eval_algorithm[eval_algorithm_name]['model'](**eval_algorithm[eval_algorithm_name]['params'])
        model.set_name(eval_algorithm_name)
        selected_arms, rewards, _, _ = evaluate_one_alg(ipw_reward[alg_idx],
                                                        n_arms,
                                                        T_timespan,
                                                        model,
                                                        output_all_arm_prob=True)

        selected_arm_all[alg_idx] = selected_arms
        rewards_all[alg_idx] = np.array(rewards)

    # After the simulation is done, increment the counter.
    with lock:
        counter.value += 1
        print(f"Job {sim_idx} done, {counter.value}/{n_simulations*repeat} completed.")

    return selected_arm_all, rewards_all


if __name__ == '__main__':
    is_print = True
    start_time = time.time()

    device = torch.device('cpu')
    config_name = sys.argv[1]
    message(f"Read configuration from {config_name}.", is_print)
    with open(config_name, 'r') as f:
        config = json.load(f)
    f.close()
    simulations_per_round = config["algorithms"]["0"]["params"]["simulation_rounds"]

    environment = config["environment"]
    env_reward = environment["reward"]
    test_case = environment['test case']
    n_simulations = environment['n_simulations']

    T_timespan = environment["base"]['T_timespan']
    n_arms = environment["base"]['n_arms']

    n_algorithms = len(config['algorithms'])
    algorithms_name = [config["algorithms"][key]["name"] for key in config["algorithms"]]
    exclude_alg = ['KL-MS+JefferysPrior', 'MS+']

    simulation_data = get_filename(T_timespan, n_simulations, test_case,
                                   simulations_per_round,
                                   is_simulation=True)
    simulation_path = 'data/' + simulation_data
    message(f'Read simulation results from {simulation_path}.', is_print)
    with open(simulation_path, 'rb') as file:
        records = pickle.load(file)
    file.close()

    eval_algorithm = {'Uniform':
                          {'model': Uniform,
                           'params': {"n_arms": n_arms, "T_timespan": T_timespan}}}
    eval_algorithms_name = list(eval_algorithm.keys())[0]

    # Use a maximum of 20 processes or the available CPU threads, whichever is smaller
    num_processes = min(20, cpu_count())
    pool = Pool(processes=num_processes)

    # Create a shared counter and a lock.
    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()

    repeat = 1
    # Start the pool with the modified function.
    eval_result = pool.starmap(evaluate_single_simulation,
                               [(i, counter, lock, repeat,
                                 records[int(i/repeat)], eval_algorithm, eval_algorithms_name,
                                 n_simulations, n_algorithms, algorithms_name, T_timespan, n_arms, env_reward)
                                for i in range(n_simulations*repeat)])

    print(f"All {n_simulations} evaluations completed.")

    pool.close()
    pool.join()

    filename = get_filename(T_timespan, n_simulations, test_case,
                            simulations_per_round, is_evaluation=True)
    with open(filename, 'wb') as file:
        pickle.dump(eval_result, file)
    file.close()

    message(f"Start evaluating the algorithms", is_print)

    generate_eval_plots(filename, env_reward, algorithms_name,
                        n_simulations, n_arms, n_algorithms, T_timespan,
                        ref_alg=None, exclude_alg=exclude_alg, is_print=True)
