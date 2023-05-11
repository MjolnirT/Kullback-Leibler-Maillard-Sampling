import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.cm import get_cmap


def plot_regret(regret, title=None, label=None):
    cum_regret = np.cumsum(np.array(regret))
    plt.plot(range(len(cum_regret)), cum_regret, label=label)
    plt.xlabel('Round')
    plt.ylabel('Cumulative regret')
    plt.title(title)
    plt.show()

    plt.plot(range(len(regret)), regret / np.arange(1, len(regret) + 1), label=label)
    plt.xlabel('Round')
    plt.ylabel('Average Regret')
    plt.title(title)
    plt.show()


def plot_regrets(regrets, ci=0.95, y_label=None, title=None, label=None, ref_alg=None):
    n_simulations, num_algorithm, T_timespan = regrets.shape
    average_regret = regrets.mean(axis=0)

    # calculate the confidence interval
    confidence_level = ci
    lower_bound = np.quantile(regrets, (1 - confidence_level) / 2, axis=0)
    upper_bound = np.quantile(regrets, 1 - (1 - confidence_level) / 2, axis=0)

    # Get a colormap with the number of algorithms
    cmap = get_cmap('tab10', num_algorithm)

    for idx_alg in range(num_algorithm):
        plt.plot(range(1, T_timespan + 1), average_regret[idx_alg, :], label=label[idx_alg])

        if label[idx_alg] in ref_alg:
            plt.fill_between(range(1, T_timespan + 1),
                             lower_bound[idx_alg, :],
                             upper_bound[idx_alg, :],
                             color=cmap(idx_alg),
                             alpha=0.2)
            # Add label to the confidence interval
            plt.text(
                T_timespan, upper_bound[idx_alg, -1],
                f'{label[idx_alg]}:{confidence_level}% CI', ha='right', va='bottom',
                color=cmap(idx_alg), fontsize=8
            )

    plt.xlabel('time step')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_arm_prob(arm_prob_list, title=None, label=None):
    for idx, arm_prob in enumerate(arm_prob_list):
        plt.plot(range(len(arm_prob)), arm_prob, label=label[idx])
    plt.xlabel('arm index')
    plt.ylabel('Probability of arms')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_density(rewards, title=None, label=None):
    fig, ax = plt.subplots()

    n_simulations, num_algorithm = rewards.shape

    # Iterate over the columns of the data and plot the density function for each
    for idx_alg in range(num_algorithm):
        alg_reward = rewards[:, idx_alg]

        # Fit a probability distribution to the column data
        dist = gaussian_kde(alg_reward)

        # Evaluate the PDF at a range of x-values
        x = np.linspace(alg_reward.min(), alg_reward.max(), 100)
        y = dist.pdf(x)

        ax.plot(x, y, label=label[idx_alg])

    # Add a legend and labels to the plot
    ax.legend()
    ax.set_xlabel('Evaluation reward')
    ax.set_ylabel('Frequency')
    ax.set_title(title)

    # Show the plot
    plt.show()


def binomial_KL_divergence(a, b):
    if b == 1:
        b = 1 - np.finfo(float).eps
    if b == 0:
        b = np.finfo(float).eps
    if a == 1:
        a = 1 - np.finfo(float).eps
    if a == 0:
        a = np.finfo(float).eps

    KL = a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
    if np.isnan(KL):
        print('KL is nan', KL)
        print(f'a: {a}, b: {b}')
    return KL


def gaussian_KL_divergence(mu1, sigma1, mu2, sigma2):
    KL = np.log(sigma2 / sigma1) + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2) - 0.5
    return KL


def message(message, print_flag=False):
    if print_flag:
        print(message)
