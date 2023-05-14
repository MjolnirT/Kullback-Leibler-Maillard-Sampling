import numpy as np


def simulate_one_alg(reward_probabilities, n_rounds, algorithm, output_all_arm_prob=False):
    n_arms = len(reward_probabilities)
    rewards = []
    selected_arms = []
    best_reward = np.max(reward_probabilities) * np.ones(n_rounds)
    arm_probs = np.zeros(shape=[n_rounds, n_arms])

    for t in range(n_rounds):
        chosen_arm = None
        # In the first n_arms rounds, play each arm once
        if t < n_arms:
            chosen_arm = t

        # After the first n_arms rounds, use the algorithm to select an arm
        if t >= n_arms:
            chosen_arm = algorithm.select_arm()
        # # EXPERIMENT 2: Mannually set the probability of each arm
        # if t >= n_arms:
        #     chosen_arm = 0

        # sample a reward from a Bernoulli distribution
        # reward = np.random.binomial(1, reward_probabilities[chosen_arm])
        reward = reward_probabilities[chosen_arm]
        algorithm.update(chosen_arm, reward)

        # EXPERIMENT 1: After the first n round,
        # assign the probability of the optimal arm with 0.9 and the rest with 0.1
        # if t == n_arms:
        #     arm_probs[t] = np.array([0.1] * n_arms)
        #     arm_probs[t, np.argmax(reward_probabilities)] = 0.9
        #     algorithm.set_arm_prob(arm_probs[t])

        # record the results
        selected_arms.append(chosen_arm)
        rewards.append(reward)

        # record the probability of each arm
        if output_all_arm_prob:
            arm_probs[t] = algorithm.get_arm_prob()

    arm_probs[-1] = algorithm.get_arm_prob()

    return selected_arms, rewards, best_reward, arm_probs


def simulate_single_simulation(simulation_idx, counter, lock, algorithms, algorithms_name, T_timespan, n_simulations, n_arms, reward_probabilities):
    selected_arm_all = np.zeros(shape=[len(algorithms), T_timespan])
    regrets_all = np.zeros(shape=[len(algorithms), T_timespan])
    arm_probs_all = np.zeros(shape=[len(algorithms), T_timespan, n_arms])
    expected_rewards_all = np.zeros(shape=[len(algorithms)])

    for alg_idx, (algorithm, args) in enumerate(algorithms):
        model = algorithm(*args)
        model.set_name(algorithms_name[alg_idx])
        selected_arms, rewards, best_reward, arm_prob = simulate_one_alg(reward_probabilities,
                                                                         T_timespan,
                                                                         model,
                                                                         output_all_arm_prob=True)

        selected_arm_all[alg_idx] = selected_arms
        regrets_all[alg_idx] = np.array(best_reward) - np.array(rewards)
        arm_probs_all[alg_idx] = arm_prob
        expected_rewards_all[alg_idx] = np.array(reward_probabilities).dot(arm_probs_all[alg_idx, -1, :])

    # After the simulation is done, increment the counter.
    with lock:
        counter.value += 1
        print(f"Job {simulation_idx} done, {counter.value}/{n_simulations} completed.")

    return selected_arm_all, regrets_all, arm_probs_all, expected_rewards_all