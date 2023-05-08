import matplotlib.pyplot as plt
import numpy as np


def plot_regret(regret, title):
    cum_regret = np.cumsum(np.array(regret))
    plt.plot(range(len(cum_regret)), cum_regret)
    plt.xlabel('Round')
    plt.ylabel('Cumulative regret')
    plt.title(title)
    plt.show()

    plt.plot(range(len(regret)), regret/np.arange(1, len(regret)+1))
    plt.xlabel('Round')
    plt.ylabel('Average Regret')
    plt.title(title)
    plt.show()


def plot_regrets(regrest_list, title):
    for regret in regrest_list:
        cum_regret = np.cumsum(np.array(regret))
        plt.plot(range(len(cum_regret)), cum_regret)
    plt.xlabel('Round')
    plt.ylabel('Cumulative regret')
    plt.legend(['Bernoulli Thompson Sampling with Jeffreys Prior', 'KL-MS'])
    plt.title(title)
    plt.show()

    for regret in regrest_list:
        plt.plot(range(len(regret)), regret/np.arange(1, len(regret)+1))
    plt.xlabel('Round')
    plt.ylabel('Average Regret')
    plt.title(title)
    plt.legend(['Bernoulli Thompson Sampling with Jeffreys Prior', 'KL-MS'])
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
