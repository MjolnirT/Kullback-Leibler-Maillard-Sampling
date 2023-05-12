from utility import *
from BernoulliTS import BernoulliTS
from BernoulliKLMS import KLMS, KLMSJefferysPrior
from MS import MS, MSPlus
from simulate import simulate
from multiprocessing import Pool, cpu_count, Manager
import pickle

print_flag = True
reward_probabilities = [0.2] + [0.25]
# reward_probabilities = [0.8] + [0.9]

# to pick the best configuration for MS+, doing a grid search from 100 simulations
# opt_config = SearchOptConfig(reward_probabilities, 100)
opt_config = [2.71828183, 0.01831564, 0.36787944]

# set algorithms and their parameters
variance = 1 / 4
T_timespan = 100
n_arms = len(reward_probabilities)
n_simulations = 200
algorithms = [(BernoulliTS, [n_arms, T_timespan]),
              (KLMS, [n_arms, T_timespan]),
              (KLMSJefferysPrior, [n_arms, T_timespan]),
              (MS, [n_arms, T_timespan, 1 / 4]),
              (MSPlus, [n_arms, T_timespan] + opt_config + [1 / 4])]
algorithms_name = ['BernoulliTS', 'KLMS', 'KLMS+JefferysPrior', 'MS', 'MS+']


def simulate_single_simulation(simulation_idx, counter, lock):
    regrets = np.zeros(shape=[len(algorithms), T_timespan])
    arm_probs = np.zeros(shape=[len(algorithms), T_timespan, n_arms])
    evl_rewards = np.zeros(shape=[len(algorithms)])

    for alg_idx, (algorithm, args) in enumerate(algorithms):
        model = algorithm(*args)
        _, rewards, best_reward, arm_prob = simulate(reward_probabilities, T_timespan, model)

        regrets[alg_idx] = np.array(best_reward) - np.array(rewards)
        arm_probs[alg_idx] = arm_prob
        evl_rewards[alg_idx] = np.array(reward_probabilities).dot(arm_probs[alg_idx, -1, :])

    # After the simulation is done, increment the counter.
    with lock:
        counter.value += 1
        print(f"Job {simulation_idx} done, {counter.value}/{n_simulations} completed.")

    return regrets, evl_rewards, arm_probs


if __name__ == '__main__':

    message(f'reward_probabilities: {reward_probabilities}', print_flag=True)

    # parallel simulation process
    # Use a maximum of 16 processes or the available CPU threads, whichever is smaller
    num_processes = min(30, cpu_count())
    pool = Pool(processes=num_processes)

    # Create a shared counter and a lock.
    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()

    # Start the pool with the modified function.
    results = pool.starmap(simulate_single_simulation,
                           [(i, counter, lock) for i in range(n_simulations)])

    print(f"All {n_simulations} simulations completed.")

    # results = pool.map(simulate_single_simulation, range(n_simulations))
    pool.close()
    pool.join()

    with open('simulation.pkl', 'wb') as file:
        pickle.dump(results, file)

    with open('simulation.pkl', 'rb') as file:
        results = pickle.load(file)

    evl_rewards = np.zeros(shape=[n_simulations, len(algorithms)])
    regrets = np.zeros(shape=[n_simulations, len(algorithms), T_timespan])
    arm_probs = np.zeros(shape=[n_simulations, len(algorithms), T_timespan, n_arms])
    for i, result in enumerate(results):
        regrets[i], evl_rewards[i], arm_probs[i] = result

    plot_density(evl_rewards, 'Evaluation Reward Comparison', label=algorithms_name)

    ref_alg = "BernoulliTS"

    # plot cumulative regret vs time step
    cum_regrets = np.cumsum(regrets, axis=2)
    plot_regrets(cum_regrets,
                 ci=0.95,
                 x_label='time step',
                 y_label='Cumulative regret',
                 title='Cumulative Regret Comparison',
                 label=algorithms_name,
                 ref_alg=ref_alg,
                 add_ci=True,
                 save_path='./figures/cum_regret.png')

    plot_regrets(cum_regrets,
                 ci=0.95,
                 x_label='time step',
                 y_label='Cumulative regret',
                 title='Cumulative Regret Comparison',
                 label=algorithms_name,
                 ref_alg=ref_alg,
                 add_ci=False,
                 save_path='./figures/cum_regret_no_ci.png')

    # plot average regret vs time step
    avg_regret = cum_regrets / np.arange(1, T_timespan + 1)
    plot_regrets(avg_regret,
                 ci=0.95,
                 x_label='time step',
                 y_label='Average regret',
                 title='Average Regret Comparison',
                 label=algorithms_name,
                 ref_alg=ref_alg,
                 add_ci=True,
                 save_path='./figures/avg_regret.png')

    plot_regrets(avg_regret,
                 ci=0.95,
                 x_label='time step',
                 y_label='Average regret',
                 title='Average Regret Comparison',
                 label=algorithms_name,
                 ref_alg=ref_alg,
                 add_ci=False,
                 save_path='./figures/avg_regret_no_ci.png')

    # plot arm probability
    arm_probs_last_round = arm_probs[:, :, -1, :]
    avg_arm_prob = np.mean(arm_probs_last_round, axis=0).reshape(1, len(algorithms), n_arms)
    plot_regrets(avg_arm_prob,
                 x_label='arm index',
                 y_label='Average arm Probability',
                 title='Arm Probability Comparison',
                 label=algorithms_name,
                 save_path='./figures/arm_prob.png')

    # plot arm probability with histogram
    plot_average_arm_prob_histogram(arm_probs_last_round,
                                    bin_width=0.1,
                                    label=algorithms_name,
                                    save_path='./figures/arm_prob_hist.png')
