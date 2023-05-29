import pickle
from multiprocessing import Pool, cpu_count, Manager
from generate_eval_plots import generate_eval_plots
from Base import Uniform
from utility import *
import time


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


def evaluate_single_simulation(sim_idx, counter, lock,
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
    select_arms, regrets, arm_probs, expect_rewards = records
    rewards = np.max(env_reward) - regrets

    simulations_per_round = 1000

    padding = 1 / simulations_per_round * 0.5

    ipw_reward = np.zeros(shape=[n_algorithms, T_timespan, n_arms])
    for alg_idx in range(n_algorithms):
        for t in range(n_arms, T_timespan):
            chosen_arm = select_arms[alg_idx, t].astype(int)
            arm_prob = arm_probs[alg_idx, t, chosen_arm]
            if algorithms_name[alg_idx] == 'BernoulliTS':
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
        print(f"Job {sim_idx} done, {counter.value}/{n_simulations} completed.")

    return selected_arm_all, rewards_all


if __name__ == '__main__':
    is_print = True
    start_time = time.time()

    # env_reward = [0.2, 0.25]
    # test_case = 1

    env_reward = [0.8] + [0.9]
    test_case = 2

    # env_reward = np.linspace(0.1, 0.9, 9)
    # test_case = 3

    # env_reward = [0.8, 0.9]
    # test_case = 4

    with open('simulation_T_10000_s_2000_test2_MC_10000_p_10000_interpolation_False.pkl', 'rb') as file:
        records = pickle.load(file)
    file.close()

    n_simulations = len(records)
    n_algorithms, T_timespan, n_arms = records[0][2].shape
    eval_algorithm = {'Uniform':
                          {'model': Uniform,
                           'params': {"n_arms": n_arms, "n_rounds": T_timespan}}}
    eval_algorithms_name = list(eval_algorithm.keys())[0]
    algorithms_name = ['BernoulliTS', 'KL-MS', 'KL-MS+JefferysPrior', 'MS', 'MS+']
    # algorithms_name = ['BernoulliTS', 'KL-MS', 'KL-MS+JefferysPrior', 'MS', 'MS+', 'BernoulliTS+RiemannApprox']
    # exclude_alg = ['KL-MS+JefferysPrior', 'MS', 'MS+', 'BernoulliTS+RiemannApprox']
    exclude_alg = ['KL-MS+JefferysPrior', 'MS', 'MS+']

    # Use a maximum of 20 processes or the available CPU threads, whichever is smaller
    num_processes = min(20, cpu_count())
    pool = Pool(processes=num_processes)

    # Create a shared counter and a lock.
    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()

    # Start the pool with the modified function.
    eval_result = pool.starmap(evaluate_single_simulation,
                               [(i, counter, lock,
                                 records[i], eval_algorithm, eval_algorithms_name,
                                 n_simulations, n_algorithms, algorithms_name, T_timespan, n_arms, env_reward)
                                for i in range(n_simulations)])

    print(f"All {n_simulations} evaluations completed.")

    pool.close()
    pool.join()

    simulations_per_round = 10000
    split_points = 10000
    is_interpolation = False
    filename = get_filename(T_timespan, n_simulations, test_case,
                            simulations_per_round, split_points, is_interpolation,
                            is_evaluation=True)
    with open(filename, 'wb') as file:
        pickle.dump(eval_result, file)
    file.close()

    message(f"Start evaluating the algorithms", is_print)

    generate_eval_plots(filename, env_reward, algorithms_name,
                        n_simulations, n_arms, n_algorithms, T_timespan,
                        ref_alg=None, exclude_alg=exclude_alg, is_print=True)
