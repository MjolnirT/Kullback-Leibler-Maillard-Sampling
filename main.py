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
opt_config = [2.71828183, 0.36787944, 0.36787944]  # optimal config for 0.2, 0.25
# opt_config = [2.71828183, 0.36787944, 0.36787944]  # optimal config for 0.8, 0.9

# set algorithms and their parameters
variance = float(1 / 4)
T_timespan = 1000
n_arms = len(reward_probabilities)
n_simulations = 200
algorithms = [(BernoulliTS, [n_arms, T_timespan]),
              (KLMS, [n_arms, T_timespan]),
              (KLMSJefferysPrior, [n_arms, T_timespan]),
              (MS, [n_arms, T_timespan, variance]),
              (MSPlus, [n_arms] + [T_timespan] + opt_config + [variance])]
algorithms_name = ['BernoulliTS', 'KLMS', 'KLMS+JefferysPrior', 'MS', 'MS+']


def simulate_single_simulation(simulation_idx, counter, lock):
    selected_arm_all = np.zeros(shape=[len(algorithms), T_timespan])
    regrets_all = np.zeros(shape=[len(algorithms), T_timespan])
    arm_probs_all = np.zeros(shape=[len(algorithms), T_timespan, n_arms])
    evl_rewards_all = np.zeros(shape=[len(algorithms)])

    for alg_idx, (algorithm, args) in enumerate(algorithms):
        model = algorithm(*args)
        model.set_name(algorithms_name[alg_idx])
        selected_arms, rewards, best_reward, arm_prob = simulate(reward_probabilities,
                                                     T_timespan,
                                                     model,
                                                     output_all_arm_prob=True)

        selected_arm_all[alg_idx] = selected_arms
        regrets_all[alg_idx] = np.array(best_reward) - np.array(rewards)
        arm_probs_all[alg_idx] = arm_prob
        evl_rewards_all[alg_idx] = np.array(reward_probabilities).dot(arm_probs_all[alg_idx, -1, :])

    # After the simulation is done, increment the counter.
    with lock:
        counter.value += 1
        print(f"Job {simulation_idx} done, {counter.value}/{n_simulations} completed.")

    return selected_arm_all, regrets_all, arm_probs_all, evl_rewards_all


if __name__ == '__main__':

    message(f'reward_probabilities: {reward_probabilities}', print_flag=True)

    # parallel simulation process
    # Use a maximum of 16 processes or the available CPU threads, whichever is smaller
    num_processes = min(16, cpu_count())
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

    select_arms = np.zeros(shape=[n_simulations, len(algorithms), T_timespan])
    regrets = np.zeros(shape=[n_simulations, len(algorithms), T_timespan])
    arm_probs = np.zeros(shape=[n_simulations, len(algorithms), T_timespan, n_arms])
    evl_rewards = np.zeros(shape=[n_simulations, len(algorithms)])
    for i, result in enumerate(results):
        select_arms[i], regrets[i], arm_probs[i], evl_rewards[i] = result

    # plot_density(evl_rewards, 'Evaluation Reward Comparison', label=algorithms_name)

    # parameters for plotting
    ref_alg = "BernoulliTS"
    experiment_param = ' | mu='+str(reward_probabilities)+' | simulations='+str(n_simulations)

    # plot cumulative regret vs time step
    cum_regrets = np.cumsum(regrets, axis=2)
    plot_regrets(cum_regrets,
                 ci=0.95,
                 x_label='time step',
                 y_label='cumulative regret'+experiment_param,
                 title='Cumulative Regret Comparison'+'mu='+str(reward_probabilities),
                 label=algorithms_name,
                 ref_alg=ref_alg,
                 add_ci=True,
                 save_path='./figures/cum_regret.png')

    plot_regrets(cum_regrets,
                 ci=0.95,
                 x_label='time step',
                 y_label='cumulative regret',
                 title='Cumulative Regret Comparison'+experiment_param,
                 label=algorithms_name,
                 ref_alg=ref_alg,
                 add_ci=False,
                 save_path='./figures/cum_regret_no_ci.png')

    # plot average regret vs time step
    avg_regret = cum_regrets / np.arange(1, T_timespan + 1)
    plot_regrets(avg_regret,
                 ci=0.95,
                 x_label='time step',
                 y_label='average regret',
                 title='Average Regret Comparison'+experiment_param,
                 label=algorithms_name,
                 ref_alg=ref_alg,
                 add_ci=True,
                 save_path='./figures/avg_regret.png')

    plot_regrets(avg_regret,
                 ci=0.95,
                 x_label='time step',
                 y_label='average regret',
                 title='Average Regret Comparison'+experiment_param,
                 label=algorithms_name,
                 ref_alg=ref_alg,
                 add_ci=False,
                 save_path='./figures/avg_regret_no_ci.png')

    # plot arm probability
    arm_probs_last_round = arm_probs[:, :, -1, :]
    avg_arm_prob = np.mean(arm_probs_last_round, axis=0).reshape(1, len(algorithms), n_arms)
    plot_regrets(avg_arm_prob,
                 x_label='arm index',
                 y_label='average arm probability',
                 title='Arm Probability Comparison'+experiment_param,
                 label=algorithms_name,
                 save_path='./figures/arm_prob.png')

    # plot arm probability with histogram
    plot_average_arm_prob_histogram(arm_probs_last_round,
                                    bin_width=0.1,
                                    label=algorithms_name,
                                    x_label='arm index',
                                    y_label='average arm probability',
                                    title='Arm Probability Histogram Comparison'+experiment_param,
                                    save_path='./figures/arm_prob_hist.png')

    # plot the optimal arm probability
    arm_best = arm_probs[:, :, :, -1]
    plot_regrets(arm_best,
                 ci=0.95,
                 x_label='time step',
                 y_label='probability of the best arm',
                 title='Probability of the Best Arm Comparison'+experiment_param,
                 label=algorithms_name,
                 ref_alg="BernoulliTS",
                 add_ci=False,
                 save_path='./figures/arm_best.png')
