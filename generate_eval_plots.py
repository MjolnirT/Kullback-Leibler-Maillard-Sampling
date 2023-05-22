import pickle
import time

import numpy as np

from Base import Uniform
from utility import plot_hist_overlapped, message, plot_lines


def generate_eval_plots(filename, env_reward, algorithms_name,
                        n_simulations, n_arms, n_algorithms, T_timespan,
                        ref_alg=None, exclude_alg=None, is_print=True):
    with open(filename, 'rb') as file:
        eval_result = pickle.load(file)
    file.close()

    eval_reward = np.zeros(shape=[n_simulations, n_algorithms, T_timespan])
    select_arms = np.zeros(shape=[n_simulations, n_algorithms, T_timespan], dtype=int)
    for i in range(n_simulations):
        select_arms[i], eval_reward[i] = eval_result[i]
    eval_reward = np.cumsum(eval_reward, axis=2)

    # oracle = (np.ones(shape=n_arms) / n_arms).dot(env_reward) * T_timespan
    oracle = (np.ones(shape=n_arms) / n_arms).dot(env_reward)

    message(f"Start plotting", is_print)
    experiment_param = ' | mu=' + str(env_reward) + ' | simulations=' + str(n_simulations)
    plot_lines(eval_reward,
               ci=0.95,
               x_label='time step',
               y_label='cumulative reward',
               title='Cumulative Reward Comparison' + experiment_param,
               label=algorithms_name,
               ref_alg=ref_alg,
               add_ci=True,
               save_path='./figures/eval_reward_line.png',
               figure_size=(8,6),
               font_size=18,
               exclude_alg=exclude_alg)

    # eval_reward_last = eval_reward[:, :, -1]
    eval_reward_last = eval_reward[:, :, -1] / T_timespan

    plot_hist_overlapped(eval_reward_last,
                         x_label='average reward',
                         y_label='frequency',
                         title='Cumulative Reward Distribution' + experiment_param,
                         label=algorithms_name,
                         add_density=False,
                         oracle=oracle,
                         save_path='./figures/eval_reward_hist.png',
                         figure_size=(8, 6),
                         font_size=12,
                         exclude_alg=exclude_alg)
    MSE = np.mean((eval_reward_last - oracle) ** 2, axis=0)
    mean = np.mean(eval_reward_last, axis=0)
    message(f"MSE: {MSE}", is_print)
    message(f"mean: {mean}", is_print)

    TS_idx = 0
    KL_MS_idx = 1
    TS_approx_idx = 5
    generate_metric(eval_reward_last[:, TS_idx], oracle, is_print, algorithms_name[TS_idx])
    generate_metric(eval_reward_last[:, KL_MS_idx], oracle, is_print, algorithms_name[KL_MS_idx])
    generate_metric(eval_reward_last[:, TS_approx_idx], oracle, is_print, algorithms_name[TS_approx_idx])


def generate_metric(reward, oracle, is_print, algorithm_name):
    finite_reward = reward[np.isfinite(reward)]
    MSE = np.mean((finite_reward - oracle) ** 2)
    mean = np.mean(finite_reward)
    message(f"{algorithm_name} MSE: {MSE}", is_print)
    message(f"{algorithm_name} mean: {mean}", is_print)
    message(f'{algorithm_name} Bias: {mean - oracle}', is_print)
    message(f"{algorithm_name} Drop {len(reward) - len(finite_reward)} NaN values", is_print)
    return MSE, mean


if __name__ == '__main__':
    start_time = time.time()
    is_print = True

    env_reward = [0.8] + [0.9]
    # env_reward = np.linspace(0.1, 0.9, 9)
    with open('simulation_T_10000_s_2000_test3_MC_1000_p_20.pkl', 'rb') as file:
        records = pickle.load(file)
        file.close()

    n_simulations = len(records)
    n_algorithms, T_timespan, n_arms = records[0][2].shape
    eval_algorithm = {'Uniform':
                          {'model': Uniform,
                           'params': {"n_arms": n_arms, "n_rounds": T_timespan}}}
    eval_algorithms_name = list(eval_algorithm.keys())[0]
    algorithms_name = ['BernoulliTS', 'KL-MS', 'KL-MS+JefferysPrior', 'MS', 'MS+', 'BernoulliTSRiemann+Approx']
    exclude_alg = ['KL-MS+JefferysPrior', 'MS', 'MS+']
    filename = 'evaluation' + '_T_' + str(T_timespan) + '_s_' + str(n_simulations) + '.pkl'

    generate_eval_plots(filename, env_reward, algorithms_name,
                        n_simulations, n_arms, n_algorithms, T_timespan,
                        ref_alg=None, exclude_alg=exclude_alg, is_print=is_print)
    message(f'Time elapsed: {time.time() - start_time:.2f}s', is_print)
