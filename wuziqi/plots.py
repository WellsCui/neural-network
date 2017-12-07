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


def plot_policy_net_train_result2():
    with open('../history/logs/policy-dropout-on-layers-validate.txt', 'r') as myfile:
        data = myfile.read().replace('\n', '')
        v = np.array(eval(data))
    with open('../history/logs/policy-dropout-on-layers-accuracy.txt', 'r') as myfile:
        data = myfile.read().replace('\n', '')
        t = np.array(eval(data))

    x = np.array(range(v.shape[0]))
    t = t[x*6]
    plt.plot(x, t[:, 0], x, t[:, 1], x, t[:, 2], x, v[:, 0], x, v[:, 1], x, v[:, 2])
    plt.title('Accuracy of Policy Network')
    plt.xlabel('Iter(5e+1)')
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


plot_policy_net_train_result2()
