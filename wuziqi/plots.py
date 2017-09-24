import matplotlib.pyplot as plt
import numpy as np


def plot_policy_net_train_result():
    with open('../history/logs/policy-accuracy-validate.txt', 'r') as myfile:
        data = myfile.read().replace('\n', '')
        a = np.array(eval(data))
    x = range(a.shape[0])
    plt.plot(x, a[:, 0, 0], x, a[:, 0, 1], x, a[:, 0, 2], x, a[:, 1, 0], x, a[:, 1, 1], x, a[:, 1, 2],)
    plt.title('Accuracy of Policy Network')
    plt.xlabel('Epic(e+2)')
    plt.ylabel('Accuracy(%)')
    plt.show()


def plot_value_net_train_result():
    with open('../history/logs/value-net-validate.txt', 'r') as myfile:
        data = myfile.read().replace('\n', '')
        a = np.array(eval(data))
    x = range(a.shape[0])
    plt.plot(x, a[:, 0], x, a[:, 1])
    plt.title('Loss of Value Network')
    plt.xlabel('Epic(e+2)')
    plt.ylabel('Loss')
    plt.show()


plot_value_net_train_result()
