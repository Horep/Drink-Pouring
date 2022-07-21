import numpy as np
import matplotlib.pyplot as plt


def hat(x1, x2, x):
    return np.heaviside(x - x1, 0.5) - np.heaviside(x - x2, 0.5)


def p_gen(nodes, max_time):
    disc = np.linspace(0, max_time, nodes + 1)
    weights = np.random.uniform(0, 5, nodes)

    def p(t):
        res = 0
        for i in range(nodes):
            res += hat(disc[i], disc[i+1], t) * weights[i]
        return res
    return p
