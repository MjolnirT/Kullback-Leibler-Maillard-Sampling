import pickle
from Base import Uniform
from utility import *
import time


def evaluate_one_alg(env_reward, n_arms, n_rounds, algorithm, output_all_arm_prob=False):
    '''
    :param env_reward: numpy array with shape [n_rounds, n_arms]
    :param n_arms: scalar
    :param n_rounds: scalar
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


# def evaluate_single_simulation(simulation_idx, counter, lock, algorithms, algorithms_name, n_simulations, env_reward):
def evaluate_single_simulation(eval_algorithm, eval_algorithm_name, env_reward):
    '''
    :param eval_algorithm: dictionary. Store the algorithm and its corresponding model and parameters needed to be evaluated
    :param eval_algorithm_name: list of string. Store the name of the algorithm
    :param env_reward: numpy array with shape [number of algorithm wait to evaluate, T_timespan, n_arms]
    '''
    first_alg_key = list(eval_algorithm.keys())[0]
    T_timespan = eval_algorithm[first_alg_key]['params']['n_rounds']
    n_arms = eval_algorithm[first_alg_key]['params']['n_arms']
    n_eval_alg = env_reward.shape[0]

    selected_arm_all = np.zeros(shape=[n_eval_alg, T_timespan])
    rewards_all = np.zeros(shape=[n_eval_alg, T_timespan])

    for alg_idx in range(n_eval_alg):
        # model = algorithm(*args)
        model = eval_algorithm[eval_algorithm_name]['model'](**eval_algorithm[eval_algorithm_name]['params'])
        model.set_name(eval_algorithm_name)
        selected_arms, rewards, best_reward, arm_prob = evaluate_one_alg(env_reward[alg_idx],
                                                                         n_arms,
                                                                         T_timespan,
                                                                         model,
                                                                         output_all_arm_prob=True)

        selected_arm_all[alg_idx] = selected_arms
        rewards_all[alg_idx] = np.array(rewards)

    # After the simulation is done, increment the counter.
    # with lock:
    #     counter.value += 1
    #     print(f"Job {simulation_idx} done, {counter.value}/{n_simulations} completed.")

    return selected_arm_all, rewards_all


if __name__ == '__main__':
    is_print = True
    start_time = time.time()
    # env_reward = [0.2] + [0.25]
    env_reward = [0.8] + [0.9]
    # env_reward = np.linspace(0.1, 0.9, 9)
    with open('simulation_T_10000_s_2000_test2.pkl', 'rb') as file:
        # with open('simulation.pkl', 'rb') as file:
        results = pickle.load(file)

    n_simulations = len(results)
    n_algorithms, T_timespan, n_arms = results[0][2].shape
    select_arms = np.zeros(shape=[n_simulations, n_algorithms, T_timespan], dtype=int)
    regrets = np.zeros(shape=[n_simulations, n_algorithms, T_timespan])
    arm_probs = np.zeros(shape=[n_simulations, n_algorithms, T_timespan, n_arms])
    expect_rewards = np.zeros(shape=[n_simulations, n_algorithms])
    for i, result in enumerate(results):
        select_arms[i], regrets[i], arm_probs[i], expect_rewards[i] = result

    rewards = np.max(env_reward) - regrets
    # inverse propensity score weighting
    # T_timespan = 2000
    # n_simulations = 200
    # n_algorithms = 2
    message(f"Start constructing ipw estimator", is_print)
    ipw_reward = np.zeros(shape=[n_simulations, n_algorithms, T_timespan, n_arms])
    for sim_idx in range(n_simulations):
        for alg_idx in range(n_algorithms):
            for tim_idx in range(n_arms, T_timespan):
                chosen_arm = select_arms[sim_idx, alg_idx, tim_idx]
                ipw_reward[sim_idx, alg_idx, tim_idx, chosen_arm] = \
                    rewards[sim_idx, alg_idx, tim_idx] / \
                    arm_probs[sim_idx, alg_idx, tim_idx, chosen_arm]

    eval_reward = np.zeros(shape=[n_simulations, n_algorithms, T_timespan])
    select_arms = np.zeros(shape=[n_simulations, n_algorithms, T_timespan], dtype=int)
    eval_algorithms = {'Uniform': {'model': Uniform, 'params': {"n_arms": n_arms, "n_rounds": T_timespan}}}
    eval_algorithms_name = list(eval_algorithms.keys())[0]
    algorithms_name = ['BernoulliTS', 'KL-MS', 'KL-MS+JefferysPrior', 'MS', 'MS+']
    exclude_alg = ['KL-MS+JefferysPrior', 'MS', 'MS+']

    message(f"Start evaluating the algorithms", is_print)
    for i in range(n_simulations):
        if i % 50 == 0:
            message(f"Simulation {i}", is_print)
        select_arms[i], eval_reward[i] = evaluate_single_simulation(eval_algorithms, eval_algorithms_name,
                                                                    ipw_reward[i])
    eval_reward = np.cumsum(eval_reward, axis=2)

    message(f"Start plotting", is_print)
    experiment_param = ' | mu=' + str(env_reward) + ' | simulations=' + str(n_simulations)
    # plot_lines(eval_reward,
    #            ci=0.95,
    #            x_label='time step',
    #            y_label='cumulative reward',
    #            # title='Cumulative Reward Comparison' + experiment_param,
    #            label=algorithms_name,
    #            ref_alg="BernoulliTS",
    #            add_ci=True,
    #            save_path='./figures/eval_reward_line.png',
    #            figure_size=(8,6),
    #            font_size=18,
    #            exclude_alg=exclude_alg)

    # eval_reward_last = eval_reward[:, :, -1]
    eval_reward_last = eval_reward[:, :, -1] / T_timespan

    with open('eval_reward_last.pkl', 'wb') as file:
        pickle.dump(eval_reward_last, file)

    with open('eval_reward_last.pkl', 'rb') as file:
        eval_reward_last = pickle.load(file)
    # oracle = (np.ones(shape=n_arms) / n_arms).dot(env_reward) * T_timespan
    oracle = (np.ones(shape=n_arms) / n_arms).dot(env_reward)
    plot_hist_overlapped(eval_reward_last,
                         x_label='average reward',
                         y_label='frequency',
                         # title='Cumulative Reward Distribution' + experiment_param,
                         label=algorithms_name,
                         add_density=True,
                         oracle=oracle,
                         save_path='./figures/eval_reward_hist.png',
                         figure_size=(8, 6),
                         font_size=18,
                         exclude_alg=exclude_alg)

    # MSE = np.mean((eval_reward_last - oracle) ** 2, axis=0)
    # message(f"MSE: {MSE}", is_print)
    eval_reward_last_TS = eval_reward_last[:, 0]
    eval_reward_last_KLMS = eval_reward_last[:, 1]
    eval_reward_last_TS = eval_reward_last_TS[np.isfinite(eval_reward_last_TS)]
    eval_reward_last_KLMS = eval_reward_last_KLMS[np.isfinite(eval_reward_last_KLMS)]
    MSE_TS = np.mean((eval_reward_last_TS - oracle) ** 2)
    MSE_KL_MS = np.mean((eval_reward_last_KLMS - oracle) ** 2)
    message(f"MSE TS: {MSE_TS}", is_print)
    message(f"MSE KL-MS: {MSE_KL_MS}", is_print)
    message(f"TS Drop {len(eval_reward_last[:, 0]) - len(eval_reward_last_TS)} NaN values", is_print)
    message(f"KL-MS Drop {len(eval_reward_last[:, 1]) - len(eval_reward_last_KLMS)} NaN values", is_print)
    message(f'Time elapsed: {time.time() - start_time:.2f}s', is_print)