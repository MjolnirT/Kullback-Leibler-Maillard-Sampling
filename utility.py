import matplotlib.pyplot as plt
import numpy as np


def plot_regret(regret, title=None, label=None):
    cum_regret = np.cumsum(np.array(regret))
    plt.plot(range(len(cum_regret)), cum_regret, label=label)
    plt.xlabel('Round')
    plt.ylabel('Cumulative regret')
    plt.title(title)
    plt.show()

    plt.plot(range(len(regret)), regret/np.arange(1, len(regret)+1), label=label)
    plt.xlabel('Round')
    plt.ylabel('Average Regret')
    plt.title(title)
    plt.show()


def plot_regrets(regrest_list, title=None, label=None):
    for idx, regret in enumerate(regrest_list):
        cum_regret = np.cumsum(np.array(regret))
        plt.plot(range(len(cum_regret)), cum_regret, label=label[idx])
    plt.xlabel('Round')
    plt.ylabel('Cumulative regret')
    plt.title(title)
    plt.legend()
    plt.show()

    for idx, regret in enumerate(regrest_list):
        plt.plot(range(len(regret)), regret/np.arange(1, len(regret)+1), label=label[idx])
    plt.xlabel('Round')
    plt.ylabel('Average Regret')
    plt.title(title)
    plt.legend()
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
