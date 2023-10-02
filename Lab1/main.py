import math
import numpy as np
import matplotlib.pyplot as plt

A_BORDER = 1
B_BORDER = 2.4
GAMMA_A = np.e * np.cos(np.e)
GAMMA_B = 1 / B_BORDER * np.sin(np.exp(B_BORDER))


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
    x = np.arange(A_BORDER, B_BORDER + 0.5 * h, h)
    n = len(x)

    a = np.empty(n - 1)
    b = np.empty(n)
    c = np.empty(n - 1)
    d = np.empty(n)

    b[0] = 1 - 1 / h
    c[0] = 1 / h
    d[0] = GAMMA_A

    for i in range(1, n - 1):
        p_val = p(x[i])
        a[i - 1] = -1 / (h * h)
        b[i] = 2 / (h * h) - p_val / h + q(x[i])
        c[i] = p_val / h - 1 / (h * h)
        d[i] = f(x[i])

    a[n - 2] = 0
    b[n - 1] = 1
    d[n - 1] = GAMMA_B

    return tdma_solver(a, b, c, d)


def second_order_scheme(h):
    x = np.arange(A_BORDER, B_BORDER + 0.5 * h, h)
    n = len(x)

    a = np.empty(n - 1)
    b = np.empty(n)
    c = np.empty(n - 1)
    d = np.empty(n)

    b[0] = -3 / (2 * h) + 1
    c[0] = 2 / h
    d[0] = GAMMA_A

    for i in range(1, n - 1):
        p_val = p(x[i])
        a[i - 1] = -1 / (h * h) - p_val / (2 * h)
        b[i] = 2 / (h * h) + q(x[i])
        c[i] = -1 / (h * h) + p_val / (2 * h)
        d[i] = f(x[i])

    a[n - 2] = 0
    b[n - 1] = 1
    d[n - 1] = GAMMA_B

    alpha = h / (-2 + h * p(x[1]))
    b[0] += alpha * a[0]
    c[0] += alpha * b[1]
    d[0] += alpha * d[1]

    return tdma_solver(a, b, c, d)


def show_solutions():
    _, ax = plt.subplots(1, 2)

    x_star = np.linspace(A_BORDER, B_BORDER, 200)
    y_star = func(x_star)

    for i in range(1, 3):
        n = 10 * (i + 3)
        hs = (B_BORDER - A_BORDER) / n
        xs = np.arange(A_BORDER, B_BORDER + 0.5 * hs, hs)
        print(xs)
        ys1 = first_order_scheme(hs)
        ys2 = second_order_scheme(hs)

        ax[i-1].plot(x_star, y_star, label='function')
        ax[i-1].plot(xs, ys1, label='1st order')
        ax[i-1].plot(xs, ys2, label='2nd order')

        ax[i-1].set_title(f"h = {hs}")
        ax[i-1].legend()

    plt.show()


def error_on_step():
    errors1 = []
    errors2 = []
    steps = []
    for i in range(0, 14):
        n = 10 * 2 ** i
        hs = (B_BORDER - A_BORDER) / n
        steps.append(hs)
        xs = np.arange(A_BORDER, B_BORDER + 0.5 * hs, hs)
        y_star = func(xs)
        ys1 = first_order_scheme(hs)
        ys2 = second_order_scheme(hs)

        z1 = np.abs(y_star - ys1)
        z2 = np.abs(y_star - ys2)

        errors1.append(np.max(z1))
        errors2.append(np.max(z2))

        print(f"i = {i} calculation...")

    print("1st order slope:")
    print((np.log(errors1[-2]) - np.log(errors1[-1])) / (np.log(steps[-2]) - np.log(steps[-1])))
    print("2nd order slope:")
    print((np.log(errors2[-2]) - np.log(errors2[-1])) / (np.log(steps[-2]) - np.log(steps[-1])))

    plt.grid()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("h")
    plt.ylabel("error")

    plt.plot(steps, errors1, label='1st order')
    plt.plot(steps, errors2, label='2nd order')

    plt.legend()
    plt.show()


def rounding_error_effect():
    errors1 = []
    errors2 = []
    steps1 = []
    steps2 = []
    for i in range(0, 21):
        n = 10 * 2 ** i
        hs = (B_BORDER - A_BORDER) / n
        steps1.append(n)
        xs = np.arange(A_BORDER, B_BORDER + 0.5 * hs, hs)
        y_star = func(xs)
        ys1 = first_order_scheme(hs)

        z1 = np.abs(y_star - ys1)

        errors1.append(np.max(z1))

        print(f"i = {i} calculation...")

    for i in range(0, 17):
        n = 10 * 2 ** i
        hs = (B_BORDER - A_BORDER) / n
        steps2.append(n)
        xs = np.arange(A_BORDER, B_BORDER + 0.5 * hs, hs)
        y_star = func(xs)
        ys2 = second_order_scheme(hs)

        z2 = np.abs(y_star - ys2)

        errors2.append(np.max(z2))

        print(f"i = {i} calculation...")

    plt.grid()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("num of nodes")
    plt.ylabel("error")

    ax_min1 = steps1[np.argmin(errors1)]
    ax_min2 = steps2[np.argmin(errors2)]
    plt.plot(steps1, errors1, label='1st order')
    plt.plot(steps2, errors2, label='2nd order')
    plt.axvline(x=ax_min1, linestyle="--", color="red")
    plt.axvline(x=ax_min2, linestyle="--", color="red")

    plt.legend()

    plt.show()


rounding_error_effect()
