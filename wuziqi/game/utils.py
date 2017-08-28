import numpy as np
import tables


def partial_random(preferred_choice, random_choices, preferred_rate):
    if np.random.choice(2, p=[preferred_rate, 1 - preferred_rate]) == 0:
        return preferred_choice
    else:
        return np.random.choice(random_choices)


def create_earray(file: tables.File, name, a):
    atom = tables.Atom.from_dtype(a.dtype)
    shape = np.array(a.shape)
    shape[0] = 0
    earray = file.create_earray(file.root, name, atom, shape)
    earray.append(a)
    return earray
