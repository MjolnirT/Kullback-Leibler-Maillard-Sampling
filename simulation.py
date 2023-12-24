import time
import numpy as np
import torch
from utility import message


def simulate_one_alg(env_reward, n_arms, n_rounds, algorithm, device, output_all_arm_prob=False):
    selected_arms = torch.zeros(n_rounds, dtype=torch.int, device=device)
    rewards = torch.zeros(n_rounds, dtype=torch.float, device=device)
    # using pseudo reward to calculate the regret
    best_reward = torch.max(env_reward) * torch.ones(n_rounds, dtype=torch.float, device=device)
    
    arm_probs = torch.zeros((n_rounds, n_arms), dtype=torch.float, device=device)
    for t in range(n_rounds):
        chosen_arm = torch.NoneType
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
        selected_arms[t] = chosen_arm
        rewards[t] = return_reward
        # record the probability of each arm
        if output_all_arm_prob:
            arm_probs[t] = algorithm.get_arm_prob()

    arm_probs[-1] = algorithm.get_arm_prob()

    return selected_arms, rewards, best_reward, arm_probs


def simulate_single_simulation(simulation_idx, counter, lock, algorithms, algorithms_name, n_simulations, env_reward, device):
    first_alg_key = list(algorithms.keys())[0]
    T_timespan = torch.tensor(algorithms[first_alg_key]['params']['T_timespan'], dtype=torch.int, device=device)
    n_arms = torch.tensor(algorithms[first_alg_key]['params']['n_arms'], dtype=torch.int, device=device)

    selected_arm_all = torch.zeros((len(algorithms), T_timespan), dtype=torch.int, device=device)
    regrets_all = torch.zeros((len(algorithms), T_timespan), dtype=torch.float, device=device)
    arm_probs_all = torch.zeros((len(algorithms), T_timespan, n_arms), dtype=torch.float, device=device)
    expected_rewards_all = torch.zeros((len(algorithms)), dtype=torch.float, device=device)
    time_cost = torch.zeros((len(algorithms)), dtype=torch.float, device=device)

    for alg_idx, algorithm in enumerate(algorithms):
        start_time = time.time()
        model = algorithms[algorithm]['model'](**algorithms[algorithm]['params'])
        model.set_name(algorithms_name[alg_idx])
        selected_arms, rewards, best_reward, arm_prob = simulate_one_alg(env_reward,
                                                                         n_arms,
                                                                         T_timespan,
                                                                         model,
                                                                         device,
                                                                         output_all_arm_prob=True)
        selected_arm_all[alg_idx] = selected_arms
        regrets_all[alg_idx] = best_reward - rewards
        arm_probs_all[alg_idx] = arm_prob
        expected_rewards_all[alg_idx] = env_reward.dot(arm_probs_all[alg_idx, -1, :])
        time_cost[alg_idx] = time.time() - start_time

    # After the simulation is done, increment the counter.
    with lock:
        counter.value += 1
        print(f"Job {simulation_idx} done, {counter.value}/{n_simulations} completed.")

    return selected_arm_all, regrets_all, arm_probs_all, expected_rewards_all, time_cost


def simulate_batch_simulation(batch_size, num_batch, environment, algorithms, algorithms_name):
    batch_size = torch.tensor(batch_size).cuda()
    num_batch = torch.tensor(num_batch).cuda()
    environment = dict_to_tensor(environment)
    algorithms = dict_to_tensor(algorithms)
    algorithms_name = torch.tensor(algorithms_name).cuda()

    n_simulations = environment['n_simulations']
    env_reward = environment['reward']
    T_timespan = environment["base"]["T_timespan"]
    n_arms = environment["base"]["n_arms"]
    
    selected_arm_all = torch.zeros((n_simulations, len(algorithms), T_timespan)).cuda()
    regrets_all = torch.zeros((n_simulations, len(algorithms), T_timespan)).cuda()
    arm_probs_all = torch.zeros((n_simulations, len(algorithms), T_timespan, n_arms)).cude()
    expected_rewards_all = np.zeros((n_simulations, len(algorithms))).cuda()
    
    for batch_id in range(num_batch):
        if batch_id == num_batch - 1:
            batch_size = n_simulations - batch_id * batch_size
        
        for alg_idx, algorithms in enumerate(algorithms):
            # simulate_one_alg_batch()
            break
        break
    return



# convert a dictionary to a dictionary with all values as torch tensors
# input: a dictionary
# output: a dictionary with all values as torch tensors
def dict_to_tensor(dict_in):
    dict_out = {}
    for key, value in dict_in.items():
        if type(value) is dict:
            dict_out[key] = dict_to_tensor(value)
        else:
            dict_out[key] = torch.tensor(value).cuda()
    return dict_out