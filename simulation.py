import numpy as np


def simulate_one_alg(env_reward, n_arms, n_rounds, algorithm, output_all_arm_prob=False):
    rewards = []
    selected_arms = []
    # using pseudo reward to calculate the regret
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

        # sample a return_reward from a Bernoulli distribution
        # return_reward = np.random.binomial(1, reward_probabilities[chosen_arm])
        return_reward = env_reward[chosen_arm]
        algorithm.update(chosen_arm, return_reward)

        # record the results
        selected_arms.append(chosen_arm)
        rewards.append(return_reward)

        # record the probability of each arm
        if output_all_arm_prob:
            arm_probs[t] = algorithm.get_arm_prob()

    arm_probs[-1] = algorithm.get_arm_prob()

    return selected_arms, rewards, best_reward, arm_probs


def simulate_single_simulation(simulation_idx, counter, lock, algorithms, algorithms_name, n_simulations, env_reward):
    first_alg_key = list(algorithms.keys())[0]
    T_timespan = algorithms[first_alg_key]['params']['n_rounds']
    n_arms = algorithms[first_alg_key]['params']['n_arms']

    selected_arm_all = np.zeros(shape=[len(algorithms), T_timespan])
    regrets_all = np.zeros(shape=[len(algorithms), T_timespan])
    arm_probs_all = np.zeros(shape=[len(algorithms), T_timespan, n_arms])
    expected_rewards_all = np.zeros(shape=[len(algorithms)])

    for alg_idx, algorithm in enumerate(algorithms):
        # model = algorithm(*args)
        model = algorithms[algorithm]['model'](**algorithms[algorithm]['params'])
        model.set_name(algorithms_name[alg_idx])
        selected_arms, rewards, best_reward, arm_prob = simulate_one_alg(env_reward,
                                                                         n_arms,
                                                                         T_timespan,
                                                                         model,
                                                                         output_all_arm_prob=True)

        selected_arm_all[alg_idx] = selected_arms
        regrets_all[alg_idx] = np.array(best_reward) - np.array(rewards)
        arm_probs_all[alg_idx] = arm_prob
        expected_rewards_all[alg_idx] = np.array(env_reward).dot(arm_probs_all[alg_idx, -1, :])

    # After the simulation is done, increment the counter.
    with lock:
        counter.value += 1
        print(f"Job {simulation_idx} done, {counter.value}/{n_simulations} completed.")

    return selected_arm_all, regrets_all, arm_probs_all, expected_rewards_all
