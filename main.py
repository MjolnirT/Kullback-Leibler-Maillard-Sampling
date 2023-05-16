from SearchOptConfig import SearchOptConfig
from generate_plots import generate_plots
from utility import *
from BernoulliTS import BernoulliTS
from BernoulliKLMS import KLMS, KLMSJefferysPrior
from MS import MS, MSPlus
from simulation import simulate_single_simulation
from multiprocessing import Pool, cpu_count, Manager
import pickle

if __name__ == '__main__':
    print_flag = True
    # env_reward = [0.2] + [0.25]

    env_reward = [0.8] + [0.9]

    # to pick the best configuration for MS+, doing a grid search from 100 simulations
    opt_config = SearchOptConfig(env_reward, n_arms=len(env_reward), n_rounds=100)

    message(f'reward_probabilities: {env_reward}', print_flag=True)

    # set algorithms and their parameters
    variance = float(1 / 4)
    T_timespan = 100
    n_arms = 2
    n_simulations = 20

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
    num_processes = min(24, cpu_count())
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

    generate_plots(filename, env_reward, algorithms_name, ref_alg='BernoulliTS')

    # select_arms = np.zeros(shape=[n_simulations, len(algorithms), T_timespan])
    # regrets = np.zeros(shape=[n_simulations, len(algorithms), T_timespan])
    # arm_probs = np.zeros(shape=[n_simulations, len(algorithms), T_timespan, n_arms])
    # evl_rewards = np.zeros(shape=[n_simulations, len(algorithms)])
    # for i, result in enumerate(results):
    #     select_arms[i], regrets[i], arm_probs[i], evl_rewards[i] = result
    #
    # # plot_density(evl_rewards, 'Evaluation Reward Comparison', label=algorithms_name)
    #
    # # parameters for plotting
    # ref_alg = "BernoulliTS"
    # experiment_param = ' | mu=' + str(env_reward) + ' | simulations=' + str(n_simulations)
    #
    # # plot cumulative regret vs time step
    # cum_regrets = np.cumsum(regrets, axis=2)
    # plot_regrets(cum_regrets,
    #              ci=0.95,
    #              x_label='time step',
    #              y_label='cumulative regret' + experiment_param,
    #              title='Cumulative Regret Comparison' + 'mu=' + str(env_reward),
    #              label=algorithms_name,
    #              ref_alg=ref_alg,
    #              add_ci=True,
    #              save_path='./figures/cum_regret.png')
    #
    # plot_regrets(cum_regrets,
    #              ci=0.95,
    #              x_label='time step',
    #              y_label='cumulative regret',
    #              title='Cumulative Regret Comparison' + experiment_param,
    #              label=algorithms_name,
    #              ref_alg=ref_alg,
    #              add_ci=False,
    #              save_path='./figures/cum_regret_no_ci.png')
    #
    # # plot average regret vs time step
    # avg_regret = cum_regrets / np.arange(1, T_timespan + 1)
    # plot_regrets(avg_regret,
    #              ci=0.95,
    #              x_label='time step',
    #              y_label='average regret',
    #              title='Average Regret Comparison' + experiment_param,
    #              label=algorithms_name,
    #              ref_alg=ref_alg,
    #              add_ci=True,
    #              save_path='./figures/avg_regret.png')
    #
    # plot_regrets(avg_regret,
    #              ci=0.95,
    #              x_label='time step',
    #              y_label='average regret',
    #              title='Average Regret Comparison' + experiment_param,
    #              label=algorithms_name,
    #              ref_alg=ref_alg,
    #              add_ci=False,
    #              save_path='./figures/avg_regret_no_ci.png')
    #
    # # plot arm probability
    # arm_probs_last_round = arm_probs[:, :, -1, :]
    # avg_arm_prob = np.expand_dims(np.mean(arm_probs_last_round, axis=0), axis=0)
    # plot_regrets(avg_arm_prob,
    #              x_label='arm index',
    #              y_label='average arm probability',
    #              title='Arm Probability Comparison' + experiment_param,
    #              label=algorithms_name,
    #              save_path='./figures/arm_prob.png')
    #
    # # plot arm probability with histogram
    # plot_average_arm_prob_histogram(arm_probs_last_round,
    #                                 bin_width=0.1,
    #                                 label=algorithms_name,
    #                                 x_label='arm index',
    #                                 y_label='average arm probability',
    #                                 title='Arm Probability Histogram Comparison' + experiment_param,
    #                                 save_path='./figures/arm_prob_hist.png')
    #
    # # plot the optimal arm probability
    # arm_best = arm_probs[:, :, :, -1]
    # plot_regrets(arm_best,
    #              ci=0.95,
    #              x_label='time step',
    #              y_label='probability of the best arm',
    #              title='Probability of the Best Arm Comparison' + experiment_param,
    #              label=algorithms_name,
    #              ref_alg="BernoulliTS",
    #              add_ci=False,
    #              save_path='./figures/arm_best.png')
