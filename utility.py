import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from scipy import stats
from matplotlib.cm import get_cmap

figure_size = (8, 6)
font_size = 12


def plot_regret(regret, title=None, label=None):
    plt.figure()
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


def plot_lines(regrets, ci=0.95, x_label=None, y_label=None, title=None,
               label=None, ref_alg=None, add_ci=False, save_path=None, exclude_alg=None,
               font_size=10, figure_size=(10, 6)):
    plt.figure(figsize=figure_size)
    n_simulations, num_algorithm, T_timespan = regrets.shape
    average_regret = regrets.mean(axis=0)

    # calculate the confidence interval
    confidence_level = ci
    lower_bound = np.quantile(regrets, (1 - confidence_level) / 2, axis=0)
    upper_bound = np.quantile(regrets, 1 - (1 - confidence_level) / 2, axis=0)

    # Get a colormap with the number of algorithms
    cmap = get_cmap('tab10', num_algorithm)

    ax1 = plt.gca()  # Get the current axes

    for idx_alg in range(num_algorithm):

        if exclude_alg is not None and label[idx_alg] in exclude_alg:
            continue

        ax1.plot(range(1, T_timespan + 1), average_regret[idx_alg, :], color=cmap(idx_alg), label=label[idx_alg])

        # Add confidence interval
        if add_ci is True:
            if label[idx_alg] in ref_alg:
                print("using ", label[idx_alg], "as the reference algorithm")
                ax1.fill_between(range(1, T_timespan + 1),
                                 lower_bound[idx_alg, :],
                                 upper_bound[idx_alg, :],
                                 color=cmap(idx_alg),
                                 alpha=0.2)
                # Add label to the confidence interval
                ax1.text(
                    T_timespan, upper_bound[idx_alg, -1],
                    f'{label[idx_alg]}:{confidence_level}% CI', ha='right', va='bottom',
                    color=cmap(idx_alg), fontsize=font_size
                )

    # ax2 = ax1.twinx()  # Create a twin axes object
    # for idx_alg in range(num_algorithm):
    #     ax2.plot(range(1, T_timespan + 1), average_regret[idx_alg, :], label=label[idx_alg])

    ax1.set_xlabel(x_label, fontsize=font_size)
    ax1.set_ylabel(y_label, fontsize=font_size)
    ax1.set_title(title)
    ax1.legend(fontsize=font_size)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def plot_arm_prob(arm_probs, title=None, label=None):
    plt.figure()
    for idx, arm_prob in enumerate(arm_probs):
        plt.plot(range(len(arm_prob)), arm_prob, label=label[idx])
    plt.xlabel('arm index')
    plt.ylabel('Probability of arms')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_histogram_with_bins(arm_probs, bin_width=0.2, x_label=None, y_label=None,
                             label=None, title=None, confidence=0.95,
                             save_path=None, exclude_alg=None):
    # Calculate the average arm probability along the simulation axis
    average_probs = np.mean(arm_probs, axis=0)
    std_probs = np.std(arm_probs, axis=0)

    # Get the number of arms and models
    n_algorithms, n_arms = average_probs.shape
    n_exclude = len(exclude_alg) if exclude_alg is not None else 0

    # Plotting the histograms
    plt.figure()

    # Iterate over each model
    bin_center_offset = 0
    for alg_idx in range(n_algorithms):
        if exclude_alg is not None and label[alg_idx] in exclude_alg:
            bin_center_offset += 1
            continue
        # Get the average probabilities and standard deviations for the current model
        model_avg_probs = average_probs[alg_idx, :]
        model_std_probs = std_probs[alg_idx, :]

        # Calculate the confidence interval for each bin
        conf_interval = stats.t.interval(confidence, df=arm_probs.shape[0] - 1, loc=model_avg_probs,
                                         scale=model_std_probs / np.sqrt(arm_probs.shape[0]))

        # Calculate the center of each bin for the current model
        bin_centers = np.arange(n_arms) + (alg_idx - bin_center_offset - (n_algorithms - n_exclude - 1) / 2) * bin_width

        # Plot the histogram with error bars for the current model
        plt.bar(bin_centers, model_avg_probs, width=bin_width, alpha=0.7,
                label=f'{label[alg_idx]}' if label else f'Model {alg_idx + 1}')

        # Add error bars representing the confidence interval
        plt.errorbar(bin_centers, model_avg_probs, yerr=np.abs(conf_interval - model_avg_probs), fmt='none',
                     color='black')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(np.arange(n_arms), np.arange(1, n_arms + 1))  # Set integer indices on x-axis
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def plot_density(rewards, title=None, label=None, x_label=None, y_label=None, save_path=None):
    fig = plt.figure(figsize=figure_size)

    n_simulations, num_algorithm = rewards.shape

    # Iterate over the columns of the data and plot the density function for each
    for idx_alg in range(num_algorithm):
        alg_reward = rewards[:, idx_alg]

        # Fit a probability distribution to the column data
        dist = gaussian_kde(alg_reward)

        # Evaluate the PDF at a range of x-values
        x = np.linspace(alg_reward.min(), alg_reward.max(), 100)
        y = dist.pdf(x)

        plt.plot(x, y, label=label[idx_alg])

    # Add a legend and labels to the plot
    plt.legend(fontsize=font_size)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    # Show the plot
    plt.show()


def plot_hist_overlapped(data, title=None, label=None, x_label=None, y_label=None,
                         save_path=None, add_density=False, oracle=None, exclude_alg=None,
                         figure_size=(8, 6), font_size=12):
    fig = plt.figure(figsize=figure_size)
    n_simulations, num_algorithm = data.shape

    # Get a colormap with the number of algorithms
    cmap = get_cmap('tab10', num_algorithm)

    # Iterate over the columns of the data and plot the density function for each
    for idx_alg in range(num_algorithm):
        if exclude_alg is not None and label[idx_alg] in exclude_alg:
            continue
        alg_reward = data[:, idx_alg]
        finite_values = alg_reward[np.isfinite(alg_reward)]
        plt.hist(finite_values, bins=40, color=cmap(idx_alg), alpha=0.5, density=True, label=label[idx_alg])

    if oracle is not None:
        plt.axvline(x=oracle, color='black', linestyle='--', label='Oracle')

    if add_density:
        # Iterate over the columns of the data and plot the density function for each
        for idx_alg in range(num_algorithm):
            if exclude_alg is not None and label[idx_alg] in exclude_alg:
                continue
            alg_reward = data[:, idx_alg]
            finite_values = alg_reward[np.isfinite(alg_reward)]
            # Fit a probability distribution to the column data
            dist = gaussian_kde(finite_values)

            # Evaluate the PDF at a range of x-values
            x = np.linspace(finite_values.min(), finite_values.max(), 100)
            y = dist.pdf(x)

            plt.plot(x, y, color=cmap(idx_alg), label=label[idx_alg])

            plt.axvline(x=finite_values.mean(),
                        color=cmap(idx_alg), alpha=0.7, linestyle='--',
                        label=label[idx_alg])

    # Add a legend and labels to the plot
    plt.legend(fontsize=font_size)
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.title(title, fontsize=font_size)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
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
    return KL


def gaussian_KL_divergence(mu1, var1, mu2, var2):
    KL = np.log(np.sqrt(var2) / np.sqrt(var1)) + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5
    return KL


def message(message, print_flag=False):
    if print_flag:
        print(message)
