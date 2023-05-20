from utility import *
import pickle


def generate_plots(filename, env_reward, algorithms_name, ref_alg=None, exclude_alg=None):
    with open(filename, 'rb') as file:
        results = pickle.load(file)

    n_simulations = len(results)

    n_algorithms, T_timespan, n_arms = results[0][2].shape

    select_arms = np.zeros(shape=[n_simulations, n_algorithms, T_timespan])
    regrets = np.zeros(shape=[n_simulations, n_algorithms, T_timespan])
    arm_probs = np.zeros(shape=[n_simulations, n_algorithms, T_timespan, n_arms])
    evl_rewards = np.zeros(shape=[n_simulations, n_algorithms])
    for i, result in enumerate(results):
        select_arms[i], regrets[i], arm_probs[i], evl_rewards[i] = result

    # parameters for plotting
    experiment_param = ' | mu=' + str(env_reward) + ' | simulations=' + str(n_simulations)

    # plot cumulative regret vs time step with confidence interval
    cum_regrets = np.cumsum(regrets, axis=2)
    plot_lines(cum_regrets,
               ci=0.95,
               x_label='time step',
               y_label='cumulative regret',
               title='Cumulative Regret Comparison' + experiment_param,
               label=algorithms_name,
               ref_alg=ref_alg,
               add_ci=True,
               save_path='./figures/cum_regret.png',
               exclude_alg=exclude_alg)

    # plot cumulative regret vs time step without confidence interval
    plot_lines(cum_regrets,
               ci=0.95,
               x_label='time step',
               y_label='cumulative regret',
               title='Cumulative Regret Comparison' + experiment_param,
               label=algorithms_name,
               ref_alg=ref_alg,
               add_ci=False,
               save_path='./figures/cum_regret_no_ci.png',
               exclude_alg=exclude_alg)

    # plot average regret vs time step with confidence interval
    avg_regret = cum_regrets / np.arange(1, T_timespan + 1)
    plot_lines(avg_regret,
               ci=0.95,
               x_label='time step',
               y_label='average regret',
               title='Average Regret Comparison' + experiment_param,
               label=algorithms_name,
               ref_alg=ref_alg,
               add_ci=True,
               save_path='./figures/avg_regret.png',
               exclude_alg=exclude_alg)

    # plot average regret vs time step without confidence interval
    plot_lines(avg_regret,
               ci=0.95,
               x_label='time step',
               y_label='average regret',
               title='Average Regret Comparison' + experiment_param,
               label=algorithms_name,
               ref_alg=ref_alg,
               add_ci=False,
               save_path='./figures/avg_regret_no_ci.png',
               exclude_alg=exclude_alg)

    # plot arm probability vs arm index
    arm_probs_last_round = arm_probs[:, :, -1, :]
    avg_arm_prob = np.expand_dims(np.mean(arm_probs_last_round, axis=0), axis=0)
    plot_lines(avg_arm_prob,
               x_label='arm index',
               y_label='average arm probability',
               title='Arm Probability Comparison' + experiment_param,
               label=algorithms_name,
               save_path='./figures/arm_prob.png',
               exclude_alg=exclude_alg)

    # plot arm probability in histogram
    plot_histogram_with_bins(arm_probs_last_round,
                             bin_width=0.1,
                             label=algorithms_name,
                             x_label='arm index',
                             y_label='average arm probability',
                             title='Arm Probability Histogram Comparison' + experiment_param,
                             save_path='./figures/arm_prob_hist.png',
                             exclude_alg=exclude_alg)

    # plot the optimal arm probability vs time step
    arm_best = arm_probs[:, :, :, -1]
    plot_lines(arm_best,
               ci=0.95,
               x_label='time step',
               y_label='probability of the best arm',
               title='Probability of the Best Arm Comparison' + experiment_param,
               label=algorithms_name,
               ref_alg="BernoulliTS",
               add_ci=False,
               save_path='./figures/arm_best.png',
               exclude_alg=exclude_alg)


if __name__ == '__main__':
    filename = 'simulation_T_10000_s_2000_test2.pkl'
    env_reward = [0.8] + [0.9]
    algorithms_name = ['BernoulliTS', 'KLMS', 'KLMS+JefferysPrior', 'MS', 'MS+']
    ref_alg = ["MS", 'BernoulliTS']
    exclude_alg = ['KLMS+JefferysPrior','MS+']
    generate_plots(filename, env_reward, algorithms_name, ref_alg, exclude_alg)
