import numpy as np


def partial_random(preferred_choice, random_choices, preferred_rate):
    if np.random.choice(2, p=[preferred_rate, 1 - preferred_rate]) == 0:
        return preferred_choice
    else:
        return np.random.choice(random_choices)
