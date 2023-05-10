from utility import *
from BernoulliTS import BernoulliTS
from BernoulliKLMS import KLMS, KLMSJefferysPrior
from MS import MS, MSPlus
from SearchOptConfig import SearchOptConfig
from simulate import simulate


if __name__ == '__main__':
    reward_probabilities = [0.1] * 5 + [0.5] * 10 + [0.8] * 10 + [0.97] * 10 + [0.98] * 5 + [0.999]
    # reward_probabilities = [0.1]*5 + [0.5]*10 + [0.6]

    print_flag = True
    message(f'reward_probabilities: {reward_probabilities}', print_flag=True)
    n_rounds = 100
    n_arms = len(reward_probabilities)

    # doing simulation for 100 times

    n_simulations = 100
    opt_config = SearchOptConfig(reward_probabilities, n_rounds)
    variance = 1/4

    algorithms = [(BernoulliTS, [n_arms, n_rounds]),
                  (KLMS, [n_arms, n_rounds]),
                  (KLMSJefferysPrior, [n_arms, n_rounds]),
                  (MS, [n_arms, n_rounds, 1/4]),
                  (MSPlus, [n_arms, n_rounds] + list(opt_config) + [1/4])]
    algorithms_name = ['BernoulliTS', 'KLMS', 'KLMS+JefferysPrior', 'MS', 'MS+']

    avg_reward = np.zeros(shape=[n_simulations, len(algorithms)])
    for i in range(n_simulations):
        if i % 10 == 0:
            message(f'Simulation {i}', print_flag=print_flag)
        for alg_idx, (algorithm, args) in enumerate(algorithms):
            model = algorithm(*args)
            _, rewards, best_reward = simulate(reward_probabilities, n_rounds, model)
            regret = np.array(best_reward) - np.array(rewards)

            arm_prob = model.get_arm_prob()
            avg_reward[i, alg_idx] = np.array(reward_probabilities).dot(arm_prob)

    plot_density(avg_reward, 'Average Reward Comparison', label=algorithms_name)

    message('Running BernoulliTS', print_flag=print_flag)
    BernoulliTS = BernoulliTS(n_arms, n_rounds)
    _, rewards, best_reward = simulate(reward_probabilities, n_rounds, BernoulliTS)
    B_TS_regret = np.array(best_reward) - np.array(rewards)

    message('Running KLMS', print_flag=print_flag)
    KLMS = KLMS(n_arms, explore_weight=1.0, n_rounds=n_rounds)
    _, rewards, best_reward = simulate(reward_probabilities, n_rounds, KLMS)
    MS_regret = np.array(best_reward) - np.array(rewards)
    message(f"Probability of arms: {KLMS.get_arm_prob()}", print_flag=print_flag)

    message('Running KLMS with JeffreysPrior', print_flag=print_flag)
    KLMSJefferysPrior = KLMSJefferysPrior(n_arms, explore_weight=1.0, n_rounds=n_rounds)
    _, rewards, best_reward = simulate(reward_probabilities, n_rounds, KLMSJefferysPrior)
    MS_Jeff_regret_2 = np.array(best_reward) - np.array(rewards)
    message(f"Probability of arms: {KLMSJefferysPrior.get_arm_prob()}", print_flag=print_flag)

    label = ['Bernoulli TS', 'KLMS', 'KLMS with Jeffreys Prior']
    regret_list = [B_TS_regret, MS_regret, MS_Jeff_regret_2]
    arm_prob_list = [BernoulliTS.get_arm_prob(), KLMS.get_arm_prob(), KLMSJefferysPrior.get_arm_prob()]

    plot_regrets(regret_list, 'Regret Comparison', label)
    plot_arm_prob(arm_prob_list, 'Arm Probability Comparison', label)