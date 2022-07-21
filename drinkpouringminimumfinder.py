import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from weightedfunctiongenerator import p_gen
# parameters
H_g = 1
H_0 = 0.8*H_g
mu = 2
k = 25
dt = 0.001
spillage_check = 0
create_animation = 0
t_max = 1

p = 0

x = np.linspace(0, t_max, 1000)


def q(t, b):
    return k*b


def spill_condition(h, b):
    if 0 <= h+b <= H_g:
        return True
    else:
        print("Spilled!")
        return False


def drink_simulate(p):
    # initial conditions
    h = 0
    b = 0
    t = 0
    H = np.array([h])
    B = np.array([b])
    T = np.array([0])
    P = [0]
    Q = [0]
    n = 1
    while h <= H_0 and t <= t_max:
        t = n*dt
        h = h + p(t)*dt
        b = max(b + (mu*p(t) - q(t, b))*dt, 0)
        if b+h >= H_g:
            b = H_g - h
        n = n + 1
        T = np.append(T, t)
        H = np.append(H, h)
        B = np.append(B, b)
        P = np.append(P, p(t))
        Q = np.append(Q, q(t, b))
    return H, B, T, P, Q


def time_taken(H, B, T, P, Q):
    if np.max(H) <= H_0:
        print("Failed to converge fast enough.")
        return np.inf
    elif np.max(H+B) >= H_g:
        print("Spilled")
        return np.inf
    else:
        return np.max(T)


def plot_results(H, B, T, P, Q):
    plt.plot(T, H, label="drink")
    plt.plot(T, H+B, label="drink+foam")
    plt.plot([np.min(T), np.max(T)], [H_0, H_0], color="black", linestyle="--")
    plt.plot([np.min(T), np.max(T)], [H_g, H_g], color="black", linestyle="--")
    plt.legend()
    plt.xlabel(r"Time $t$")
    plt.ylabel(r"Height $h$")

    plt.title(f"µ={mu}, dt={dt}")
    plt.savefig("drink_foam_graph.pdf")
    plt.show()
    plt.clf()

    plt.plot(T, P)
    plt.xlabel(r"Time $t$")
    plt.ylabel(r"Inflow $p$")
    plt.savefig("drink_in.pdf")
    plt.show()
    plt.clf()

    plt.plot(T, mu*P, label=r"$\mu\cdot p$")
    plt.plot(T, Q, label=r"$q$")
    plt.xlabel(r"Time $t$")
    plt.ylabel(r"Evaporation $q$")
    plt.savefig("evaporation.pdf")
    plt.legend()
    plt.show()
    plt.clf()


def minimiser(N):
    t_old = t_max
    p_min = 0
    for i in range(N):
        p = p_gen(20, t_old)
        H, B, T, P, Q = drink_simulate(p)
        t_new = time_taken(H, B, T, P, Q)
        if t_new < t_old:
            t_old = t_new
            p_min = p
    return p_min, H, B, T, P, Q


y, H, B, T, P, Q = minimiser(100)

plot_results(H, B, T, P, Q)
