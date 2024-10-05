import json
import os.path
import pickle
import sys
import time
from multiprocessing import Pool, cpu_count, Manager
from src.Base import Uniform
from src.utility_functions.generate_eval_plots import generate_eval_plots
from src.utility_functions.utility import *
from src.utility_functions.utility_io import get_filename
from src.global_config import DATA_DIR, PLOT_DIR, LOG_FLAG


def evaluate_one_alg(env_reward, n_arms, n_rounds, algorithm, output_all_arm_prob=False):
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

        selected_arms.append(chosen_arm)
        rewards.append(return_reward)

        if output_all_arm_prob:
            arm_probs[t] = algorithm.get_arm_prob()

    arm_probs[-1] = algorithm.get_arm_prob()
    return selected_arms, rewards, best_reward, arm_probs


def evaluate_single_simulation(sim_idx, counter, lock, repeat,
                               records, eval_algorithm, eval_algorithm_name,
                               n_simulations, n_algorithms, algorithms_name, T_timespan, n_arms, env_reward, MC_simulation_round):
    selected_arm_all = np.zeros(shape=[n_algorithms, T_timespan])
    rewards_all = np.zeros(shape=[n_algorithms, T_timespan])

    select_arms, regrets, arm_probs, expect_rewards, time_cost = records
    rewards = np.max(env_reward) - regrets
    padding = 1 / MC_simulation_round * 0.5

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
        model = eval_algorithm[eval_algorithm_name]['model'](**eval_algorithm[eval_algorithm_name]['params'])
        model.set_name(eval_algorithm_name)
        selected_arms, rewards, _, _ = evaluate_one_alg(ipw_reward[alg_idx],
                                                        n_arms,
                                                        T_timespan,
                                                        model,
                                                        output_all_arm_prob=True)

        selected_arm_all[alg_idx] = selected_arms
        rewards_all[alg_idx] = np.array(rewards)

    with lock:
        counter.value += 1
        print(f"Job {sim_idx} done, {counter.value}/{n_simulations*repeat} completed.")

    return selected_arm_all, rewards_all


if __name__ == '__main__':
    start_time = time.time()

    config_name = sys.argv[1]
    message(f"Read configuration from {config_name}.", LOG_FLAG)
    with open(config_name, 'r') as f:
        config = json.load(f)

    MC_simulation_round = find_mc_simulation_round(config)
    environment = config["environment"]
    env_reward = environment["reward"]
    test_case = environment['test case']
    n_simulations = environment['n_simulations']
    T_timespan = environment["base"]['T_timespan']
    n_arms = environment["base"]['n_arms']
    n_algorithms = len(config['algorithms'])
    algorithms_name = [config["algorithms"][key]["name"] for key in config["algorithms"]]
    exclude_alg = ['KL-MS+JefferysPrior', 'MS', 'simuBernoulliTS', 'MS+']

    simulation_filename = get_filename(T_timespan, n_simulations, test_case, MC_simulation_round, is_simulation=True)
    simulation_filepath = os.path.join(DATA_DIR, simulation_filename)
    message(f'Read simulation results from {simulation_filepath}.', LOG_FLAG)
    with open(simulation_filepath, 'rb') as file:
        records = pickle.load(file)

    eval_algorithm = {'Uniform': {'model': Uniform, 'params': {"n_arms": n_arms, "n_rounds": T_timespan}}}
    eval_algorithms_name = list(eval_algorithm.keys())[0]

    num_processes = int(cpu_count()/2)
    with Pool(processes=num_processes) as pool:
        manager = Manager()
        counter = manager.Value('i', 0)
        lock = manager.Lock()

        repeat = 1
        eval_result = pool.starmap(evaluate_single_simulation,
                               [(i, counter, lock, repeat,
                                 records[int(i/repeat)], eval_algorithm, eval_algorithms_name,
                                 n_simulations, n_algorithms, algorithms_name, T_timespan, n_arms, env_reward, MC_simulation_round)
                                for i in range(n_simulations*repeat)])

    print(f"All {n_simulations} evaluations completed.")

    evaluation_filename = get_filename(T_timespan, n_simulations, test_case, MC_simulation_round, is_evaluation=True)
    evaluation_filepath = os.path.join(DATA_DIR, evaluation_filename)
    with open(evaluation_filepath, 'wb') as file:
        pickle.dump(eval_result, file)

    message(f"Start evaluating the algorithms", LOG_FLAG)

    generate_eval_plots(evaluation_filepath, env_reward, algorithms_name,
                        n_simulations, n_arms, n_algorithms, T_timespan,
                        PLOT_DIR,
                        ref_alg=None, exclude_alg=exclude_alg, is_print=LOG_FLAG)
    