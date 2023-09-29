import math
import numpy as np
import matplotlib.pyplot as plt

A_BORDER = 1
B_BORDER = 3
GAMMA_A = np.e * np.cos(np.e)  # np.sin(np.exp(1))
GAMMA_B = 1 / 3 * np.sin(np.exp(3))


def func(x):
    return np.sin(np.exp(x)) / x


def p(x):
    return -2 / x


def q(x):
    return math.exp(2 * x)


def f(x):
    exp = math.exp(x)
    return exp / x * (2 * exp * math.sin(exp) - math.cos(exp))


def tdma_solver(a, b, c, d):
    nf = len(d)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc


def first_order_scheme(h):
    x = np.arange(A_BORDER, B_BORDER + h, h)

    a = np.array([])
    b = np.array([1-1/h])
    c = np.array([1/h])
    d = np.array([GAMMA_A])

    for x_i in x[1:-1]:
        a = np.append(a, -1 / (h * h))
        b = np.append(b, 2 / (h * h) - p(x_i) / h + q(x_i))
        c = np.append(c, p(x_i) / h - 1 / (h * h))
        d = np.append(d, f(x_i))

    a = np.append(a, 0)
    b = np.append(b, 1)
    d = np.append(d, GAMMA_B)

    return tdma_solver(a, b, c, d)


hs = 0.01
x_star = np.linspace(A_BORDER, B_BORDER, 500)
y_star = func(x_star)
xs = np.arange(A_BORDER, B_BORDER + hs, hs)
ys = first_order_scheme(hs)

fig, ax = plt.subplots()
ax.plot(x_star, y_star)
ax.plot(xs, ys)
plt.show()
