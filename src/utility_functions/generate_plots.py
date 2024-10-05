import pickle
import os
import sys
import json
from .utility import *
from ..global_config import PLOT_DIR, DATA_DIR
from .utility_io import get_filename


def generate_plots(filename, env_reward, algorithms_name, output_dir, ref_alg=None, exclude_alg=None):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    file.close()

    n_simulations = len(data)

    n_algorithms, T_timespan, n_arms = data[0][2].shape

    select_arms = np.zeros(shape=[n_simulations, n_algorithms, T_timespan])
    regrets = np.zeros(shape=[n_simulations, n_algorithms, T_timespan])
    arm_probs = np.zeros(shape=[n_simulations, n_algorithms, T_timespan, n_arms])
    evl_rewards = np.zeros(shape=[n_simulations, n_algorithms])
    time_cost = np.zeros(shape=[n_simulations, n_algorithms])
    for i, result in enumerate(data):
        select_arms[i], regrets[i], arm_probs[i], evl_rewards[i], time_cost[i] = result

    # parameters for plotting
    experiment_param = ' | mu=' + str(env_reward) + ' | simulations=' + str(n_simulations)

    # plot cumulative regret vs time step with confidence interval
    cum_regrets = np.cumsum(regrets, axis=2)
    plot_filepath = os.path.join(output_dir, 'cum_regret.png')
    plot_lines(cum_regrets,
               ci=0.95,
               x_label='time step',
               y_label='cumulative regret',
               title='Cumulative Regret Comparison' + experiment_param,
               label=algorithms_name,
               ref_alg=ref_alg,
               add_ci=True,
               save_path=plot_filepath,
               exclude_alg=exclude_alg)

    # plot cumulative regret vs time step without confidence interval
    plot_filepath = os.path.join(output_dir, 'cum_regret_no_ci.png')
    plot_lines(cum_regrets,
               ci=0.95,
               x_label='time step',
               y_label='cumulative regret',
               title='Cumulative Regret Comparison' + experiment_param,
               label=algorithms_name,
               ref_alg=ref_alg,
               add_ci=False,
               save_path=plot_filepath,
               exclude_alg=exclude_alg)

    # plot average regret vs time step with confidence interval
    avg_regret = cum_regrets / np.arange(1, T_timespan + 1)
    plot_filepath = os.path.join(output_dir, 'avg_regret.png')
    plot_lines(avg_regret,
               ci=0.95,
               x_label='time step',
               y_label='average regret',
               title='Average Regret Comparison' + experiment_param,
               label=algorithms_name,
               ref_alg=ref_alg,
               add_ci=True,
               save_path=plot_filepath,
               exclude_alg=exclude_alg)

    # plot average regret vs time step without confidence interval
    plot_filepath = os.path.join(output_dir, 'avg_regret_no_ci.png')
    plot_lines(avg_regret,
               ci=0.95,
               x_label='time step',
               y_label='average regret',
               title='Average Regret Comparison' + experiment_param,
               label=algorithms_name,
               ref_alg=ref_alg,
               add_ci=False,
               save_path=plot_filepath,
               exclude_alg=exclude_alg)

    # plot arm probability vs arm index
    arm_probs_last_round = arm_probs[:, :, -1, :]
    avg_arm_prob = np.expand_dims(np.mean(arm_probs_last_round, axis=0), axis=0)
    plot_filepath = os.path.join(output_dir, 'arm_prob.png')
    plot_lines(avg_arm_prob,
               x_label='arm index',
               y_label='average arm probability',
               title='Arm Probability Comparison' + experiment_param,
               label=algorithms_name,
               save_path=plot_filepath,
               exclude_alg=exclude_alg)

    # plot the optimal arm probability vs time step
    arm_best = arm_probs[:, :, :, -1]
    plot_filepath = os.path.join(output_dir, 'arm_best.png')
    plot_lines(arm_best,
               ci=0.95,
               x_label='time step',
               y_label='probability of the best arm',
               title='Probability of the Best Arm Comparison' + experiment_param,
               label=algorithms_name,
               ref_alg="BernoulliTS",
               add_ci=False,
               save_path=plot_filepath,
               exclude_alg=exclude_alg)

    message(f'Average time cost for each algorithm: {np.mean(time_cost, axis=0)}', True)


if __name__ == '__main__':
    config_filepath = sys.argv[1]
    with open(config_filepath, 'r') as f:
        config = json.load(f)
    env_reward = config["environment"]["reward"]
    test_case = config["environment"]["test case"]
    T_timespan = config["environment"]["base"]["T_timespan"]
    n_arms = config["environment"]["base"]["n_arms"]
    n_simulations = config["environment"]["n_simulations"]
    algorithms_name = [config["algorithms"][key]["name"] for key in config["algorithms"]]

    MC_simulation_round = find_mc_simulation_round(config)

    simulation_filename = get_filename(T_timespan, n_simulations, test_case, MC_simulation_round, is_simulation=True)
    simulation_filepath = os.path.join(DATA_DIR, simulation_filename)

    algorithms_name = ['Bernoulli Thompson Sampling', 'KL-MS', 'KLMS+JefferysPrior', 'MS', 'MS+', 'BernoulliTS+RiemannApprox']
    ref_alg = ["MS", 'KL-MS', 'Bernoulli Thompson Sampling']
    exclude_alg = ['KLMS+JefferysPrior', 'MS+']
    generate_plots(simulation_filepath, env_reward, algorithms_name, PLOT_DIR, ref_alg, exclude_alg)
