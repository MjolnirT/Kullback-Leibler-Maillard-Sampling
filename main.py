from utility import *
from BernoulliTS import BernoulliTS
from BernoulliKLMS import KLMS, KLMSJefferysPrior
from MS import MS, MSPlus
from SearchOptConfig import SearchOptConfig
from simulate import simulate
from multiprocessing import Pool, cpu_count


print_flag = True
reward_probabilities = [0.1] + [0.2]
# reward_probabilities = [0.1] * 5 + [0.5] * 10 + [0.8] * 10 + [0.97] * 10 + [0.98] * 5 + [0.999]
# reward_probabilities = [0.1]*5 + [0.5]*10 + [0.6]

# to pick the best configuration for MS+, doing a grid search from 100 simulations
opt_config = SearchOptConfig(reward_probabilities, 100)

# set algorithms and their parameters
variance = 1/4
T_timespan = 100
n_arms = len(reward_probabilities)
n_simulations = 100
algorithms = [(BernoulliTS, [n_arms, T_timespan]),
              (KLMS, [n_arms, T_timespan]),
              (KLMSJefferysPrior, [n_arms, T_timespan]),
              (MS, [n_arms, T_timespan, 1 / 4]),
              (MSPlus, [n_arms, T_timespan] + list(opt_config) + [1 / 4])]
algorithms_name = ['BernoulliTS', 'KLMS', 'KLMS+JefferysPrior', 'MS', 'MS+']


def simulate_single_simulation(simulation_idx):
    regrets = np.zeros(shape=[len(algorithms), T_timespan])
    evl_rewards = np.zeros(shape=[len(algorithms)])

    for alg_idx, (algorithm, args) in enumerate(algorithms):
        model = algorithm(*args)
        _, rewards, best_reward = simulate(reward_probabilities, T_timespan, model)
        regrets[alg_idx] = np.array(best_reward) - np.array(rewards)

        arm_prob = model.get_arm_prob()
        evl_rewards[alg_idx] = np.array(reward_probabilities).dot(arm_prob)

    return regrets, evl_rewards


if __name__ == '__main__':

    message(f'reward_probabilities: {reward_probabilities}', print_flag=True)


    evl_rewards = np.zeros(shape=[n_simulations, len(algorithms)])
    regrets = np.zeros(shape=[n_simulations, len(algorithms), T_timespan])

    num_processes = min(16, cpu_count())  # Use a maximum of 32 processes or the available CPU threads, whichever is smaller
    pool = Pool(processes=num_processes)

    results = pool.map(simulate_single_simulation, range(n_simulations))

    for i, result in enumerate(results):
        regrets[i], evl_rewards[i] = result

    pool.close()
    pool.join()

    plot_density(evl_rewards, 'Evaluation Reward Comparison', label=algorithms_name)

    ref_alg = "BernoulliTS"

    # plot cumulative regret vs time step
    cum_regrets = np.cumsum(regrets, axis=2)
    plot_regrets(cum_regrets,
                 ci=0.95,
                 y_label='Cumulative regret',
                 title='Regret Comparison',
                 label=algorithms_name,
                 ref_alg=ref_alg)

    # plot average regret vs time step
    avg_regret = cum_regrets / np.arange(1, T_timespan + 1)
    plot_regrets(avg_regret,
                 ci=0.95,
                 y_label='Average regret',
                 title='Regret Comparison',
                 label=algorithms_name,
                 ref_alg=ref_alg)

    # plot arm probability
    # plot_arm_prob(arm_prob_list, 'Arm Probability Comparison', label)