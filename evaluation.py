import pickle
from Base import Uniform
from utility import *


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


env_reward = [0.2, 0.25]
with open('simulation.pkl', 'rb') as file:
    results = pickle.load(file)

n_simulations = len(results)
n_algorithms, T_timespan, n_arms= results[0][2].shape
select_arms = np.zeros(shape=[n_simulations, n_algorithms, T_timespan], dtype=int)
regrets = np.zeros(shape=[n_simulations, n_algorithms, T_timespan])
arm_probs = np.zeros(shape=[n_simulations, n_algorithms, T_timespan, n_arms])
expect_rewards = np.zeros(shape=[n_simulations, n_algorithms])
for i, result in enumerate(results):
    select_arms[i], regrets[i], arm_probs[i], expect_rewards[i] = result

rewards = np.max(env_reward) - regrets
# inverse propensity score weighting
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
eval_algorithms = {'Uniform':
                       {'model': Uniform,
                        'params': {"n_arms": n_arms, "n_rounds": T_timespan}}}
eval_algorithms_name = list(eval_algorithms.keys())[0]
algorithms_name = ['BernoulliTS','KL-MS', 'KL-MS+JefferysPrior', 'MS', 'MS+']
for i in range(n_simulations):
    select_arms[i], eval_reward[i] = evaluate_single_simulation(eval_algorithms, eval_algorithms_name, ipw_reward[i])
eval_reward = np.cumsum(eval_reward, axis=2)

experiment_param = ' | mu=' + str(env_reward) + ' | simulations=' + str(n_simulations)
plot_regrets(eval_reward,
             ci=0.95,
             x_label='time step',
             y_label='cumulative reward',
             title='Cumulative Reward Comparison' + experiment_param,
             label=algorithms_name,
             ref_alg="BernoulliTS",
             add_ci=True)

eval_reward_last = eval_reward[:, :, -1]
oracle = arm_probs[:, :, -1, :].mean(axis=0).dot(env_reward)*T_timespan
plot_hist(eval_reward_last,
          x_label='cumulative reward',
          y_label='frequency',
          title='Cumulative Reward Distribution' + experiment_param,
          label=algorithms_name,
          add_density=True,
          oracle=oracle)
