import pickle
import numpy as np


def unpickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f, encoding="latin1")


# print(unpickle('../cifar-10-batches-py/data_batch_1'))

np.argmin([[1,2,3],[2,3,4]],0)