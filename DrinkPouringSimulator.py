import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
# parameters
H_g = 1
H_0 = 0.8*H_g
mu = 2
k = 25
dt = 0.001
spillage_check = 0
create_animation = 0

def p(t):
    return 4*t


def q(t, b):
    return k*b


# initial conditions
h = 0
b = 0
t = 0
t_max = 100
H = np.array([h])
B = np.array([b])
T = np.array([0])
P = [p(0)]
Q = [q(0, 0)]
n = 1


def spill_condition(h, b):
    if 0 <= h+b <= H_g:
        return True
    else:
        print("Spilled!")
        return False


while h <= H_0 and t <= t_max:
    t = n*dt
    h = h + p(t)*dt
    b = max(b + (mu*p(t) - q(t, b))*dt, 0)
    if b+h >= H_g:
        b = H_g - h
        spillage_check = 1
    n = n + 1
    T = np.append(T, t)
    H = np.append(H, h)
    B = np.append(B, b)
    P = np.append(P, p(t))
    Q = np.append(Q, q(t, b))


print(f"Time taken to fill = {np.max(T)}")

if spillage_check == 1:
    print("Spilled!")
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

plt.plot(T, Q)
plt.xlabel(r"Time $t$")
plt.ylabel(r"Evaporation $q$")
plt.savefig("evaporation.pdf")
plt.show()
plt.clf()

if create_animation == 1:
    for i in range(np.size(T)):
        if i % 100 == 0:
            fig, ax = plt.subplots()
            ax.set_xlim([0, 1])
            ax.set_ylim([0, H_g])
            rect1 = patches.Rectangle((0, 0), 1, H[i], linewidth=1, edgecolor='r', facecolor='saddlebrown')
            rect2 = patches.Rectangle((0, H[i]), 1, B[i], linewidth=1, edgecolor='r', facecolor='goldenrod')
            ax.add_patch(rect1)
            ax.add_patch(rect2)
            ax.title.set_text(f"t = {np.round(T[i], 2)}")
            fig.savefig(f"Animation/frame{i}.png")
            plt.close(fig)
