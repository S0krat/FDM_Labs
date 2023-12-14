import matplotlib.pyplot as plt
import numpy as np
import time

LEFT = 0
RIGHT = 1
T = 3


def exact_func(x, t):
    return (np.sin(np.pi * t + np.pi * x) * np.exp(-t) + 1) * x


def f_func(x, t):
    return np.sin(np.pi * t + np.pi * x) * np.exp(-t) * (np.pi * np.pi * x * x / 2 - x)


def rho(x):
    return x / 2


def gamma_0(t):  # left boundary condition
    return 0  # exact_func(LEFT, t)


def gamma_1(t):  # right boundary condition
    return np.sin(np.pi * (t + 1)) * np.exp(-t) + 1


def phi(x):  # lower boundary condition
    return (np.sin(np.pi * x) + 1) * x


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


def create_sol(ts, xs):
    n = len(ts)
    m = len(xs)
    sol = np.zeros((n, m))
    for i in range(m):
        sol[0, i] = phi(xs[i])

    for j in range(n):
        sol[j, 0] = gamma_0(ts[j])

    for j in range(n):
        sol[j, m - 1] = gamma_1(ts[j])

    return sol


def explicit_scheme_error(m, delta, n=-1):
    h = (RIGHT - LEFT) / (m - 1)
    if n == -1:
        tau = h * h + delta
        n = int(np.ceil(T / tau))
        tau = T / (n - 1)
    else:
        tau = T / (n - 1)
    xs = np.linspace(LEFT, RIGHT, m)
    ts = np.linspace(0, T, n)

    sol = create_sol(ts, xs)

    for j in range(n - 1):
        for i in range(1, m - 1):
            sol[j + 1, i] = tau * rho(xs[i]) / (h * h) * (sol[j, i + 1] - 2 * sol[j, i] + sol[j, i - 1]) + sol[j, i] + \
                            tau * f_func(xs[i], ts[j])

    errors = np.zeros(m)
    for i in range(m):
        func_values = np.array([exact_func(xs[i], t) for t in ts])
        errors[i] = np.max(np.abs(func_values - sol[:, i]))

    return np.max(errors) / 1.28


def explicit_scheme_plot_sols(m):
    h = (RIGHT - LEFT) / (m - 1)
    tau = h * h
    n = int(np.ceil(T / tau))
    tau = T / (n - 1)
    xs = np.linspace(LEFT, RIGHT, m)
    ts = np.linspace(0, T, n)

    sol = create_sol(ts, xs)

    for j in range(n - 1):
        for i in range(1, m - 1):
            sol[j + 1, i] = tau * rho(xs[i]) / (h * h) * (sol[j, i + 1] - 2 * sol[j, i] + sol[j, i - 1]) + sol[j, i] + \
                            tau * f_func(xs[i], ts[j])

    plt.grid()
    plt.plot(xs, xs, label="Stable", linewidth=3, color="black")
    for j in range(n // 6, n, n // 6):
        plt.plot(xs, sol[j, :], label=f"t={ts[j]:.2f}")
    plt.legend()
    plt.show()


def explicit_scheme_plot_errors():
    nodes = []
    ers = []
    for k in range(8):
        n_num = int(5 * 1.5 ** k)
        print(k, end=' : ')
        print(n_num)
        er = explicit_scheme_error(n_num, 0)
        nodes.append(RIGHT / n_num)
        ers.append(er)

    plt.grid()
    print(nodes)
    print(ers)
    plt.plot(nodes, ers)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("h")
    plt.ylabel("error")
    plt.show()


def explicit_scheme_errors_on_tau():
    errors = []
    d_taus = []
    for i in range(1, 20):
        d_tau = 1e-4 * (2 + i / 50)
        d_taus.append(d_tau)
        error = explicit_scheme_error(30, d_tau)
        errors.append(error)
        print(i)

    plt.grid()
    plt.plot(d_taus, errors)
    plt.yscale("log")
    plt.xlabel("$d\\tau$")
    plt.ylabel("error")
    plt.show()


def implicit_scheme(m, n):
    h = (RIGHT - LEFT) / (m - 1)
    tau = T / (n - 1)
    xs = np.linspace(LEFT, RIGHT, m)
    ts = np.linspace(0, T, n)

    sol = create_sol(ts, xs)

    h_sq = h * h
    h_sq_tau = h_sq / tau

    for j in range(1, n):
        a = np.empty(m - 3)
        b = np.empty(m - 2)
        c = np.empty(m - 3)
        d = np.empty(m - 2)

        r = rho(xs[1])
        b[0] = h_sq_tau + 2 * r
        c[0] = -r
        d[0] = h_sq * f_func(xs[1], ts[j]) + h_sq_tau * sol[j-1, 1] + r * sol[j, 0]

        for k in range(1, m - 3):
            r = rho(xs[k + 1])
            a[k - 1] = -r
            b[k] = h_sq_tau + 2 * r
            c[k] = -r
            d[k] = h_sq * f_func(xs[k + 1], ts[j]) + h_sq_tau * sol[j - 1, k + 1]

        r = rho(xs[m - 2])
        a[m - 4] = -r
        b[m - 3] = h_sq_tau + 2 * r
        d[m - 3] = h_sq * f_func(xs[m - 2], ts[j]) + h_sq_tau * sol[j - 1, m - 2] + r * sol[j, m - 1]

        sol[j, 1:m-1] = tdma_solver(a, b, c, d)

    return sol


def implicit_scheme_error(m, n):
    sol = implicit_scheme(m, n)
    xs = np.linspace(LEFT, RIGHT, m)
    ts = np.linspace(0, T, n)
    errors = np.zeros(m)
    for j in range(m):
        func_values = np.array([exact_func(xs[j], t) for t in ts])
        errors[j] = np.max(np.abs(func_values - sol[:, j]))
    return np.max(errors) / 1.28


def implicit_scheme_error_on_tau():
    ms = [11, 25, 56]
    errors = [0.004114340922257379, 0.000704891637241193, 0.00013398881184402758]
    colors = ["red", "green", "blue"]
    _, ax = plt.subplots(1, 3)
    for j in range(len(ms)):
        m = ms[j]
        h = (RIGHT - LEFT) / (m - 1)
        tau_0 = h * h
        taus = []
        ers = []

        for i in range(10):
            tau = tau_0 * np.sqrt(2) ** i
            taus.append(tau)
            n = int(T / tau) + 1
            print(n)
            ers.append(implicit_scheme_error(m, n))
        ax[j].grid()
        ax[j].axhline(y=errors[j], color=colors[j], linestyle="--")
        ax[j].plot(taus, ers, color=colors[j])
        ax[j].set_title(f"h={h:.3f}")
        ax[j].set_xscale("log")
        # ax[j].set_yscale("log")
        ax[j].set_xlabel("$\\tau$")

    ax[0].set_ylabel("error")
    plt.show()


def crank_nicolson(m, n):
    h = (RIGHT - LEFT) / (m - 1)
    tau = T / (n - 1)
    xs = np.linspace(LEFT, RIGHT, m)
    ts = np.linspace(0, T, n)

    sol = create_sol(ts, xs)

    h_sq = h * h
    h_sq_tau = h_sq / tau

    for j in range(1, n):
        a = np.empty(m - 3)
        b = np.empty(m - 2)
        c = np.empty(m - 3)
        d = np.empty(m - 2)

        r = rho(xs[1])
        b[0] = h_sq_tau + r
        c[0] = -r / 2
        d[0] = (h_sq * f_func(xs[1], ts[j] - tau / 2) + (h_sq_tau - r) * sol[j - 1, 1] +
                r / 2 * sol[j - 1, 0] + r / 2 * sol[j - 1, 2] + r / 2 * sol[j, 0])

        for k in range(1, m - 3):
            r = rho(xs[k + 1])
            a[k - 1] = -r / 2
            b[k] = h_sq_tau + r
            c[k] = -r / 2
            d[k] = (h_sq * f_func(xs[k + 1], ts[j] - tau / 2) + (h_sq_tau - r) * sol[j - 1, k + 1] +
                    r / 2 * sol[j - 1, k] + r / 2 * sol[j - 1, k + 2])

        r = rho(xs[m - 2])
        a[m - 4] = -r / 2
        b[m - 3] = h_sq_tau + r
        d[m - 3] = (h_sq * f_func(xs[m - 2], ts[j] - tau / 2) + (h_sq_tau - r) * sol[j - 1, m - 2] +
                    r / 2 * sol[j - 1, m - 3] + r / 2 * sol[j - 1, m - 1] + r / 2 * sol[j, m - 1])

        sol[j, 1:m - 1] = tdma_solver(a, b, c, d)

    return sol


def crank_nicolson_error(m, n):
    sol = crank_nicolson(m, n)
    xs = np.linspace(LEFT, RIGHT, m)
    ts = np.linspace(0, T, n)
    errors = np.zeros(m)
    for j in range(m):
        func_values = np.array([exact_func(xs[j], t) for t in ts])
        errors[j] = np.max(np.abs(func_values - sol[:, j]))
    return np.max(errors) / 1.28


def crank_nicolson_error_on_tau():
    ms = [11, 25, 56]
    errors = [0.004114340922257379, 0.000704891637241193, 0.00013398881184402758]
    colors = ["red", "green", "blue"]
    _, ax = plt.subplots(1, 3)
    for j in range(len(ms)):
        m = ms[j]
        h = (RIGHT - LEFT) / (m - 1)
        tau_0 = h * h
        taus = []
        ers = []

        for i in range(7 + 3 * j):
            tau = tau_0 * np.sqrt(2) ** i
            taus.append(tau)
            n = int(T / tau) + 1
            print(n)
            ers.append(crank_nicolson_error(m, n))
        ax[j].grid()
        ax[j].axhline(y=errors[j], color=colors[j], linestyle="--")
        ax[j].plot(taus, ers, color=colors[j])
        ax[j].set_title(f"h={h:.3f}")
        ax[j].set_xscale("log")
        ax[j].set_xlabel("$\\tau$")

    ax[0].set_ylabel("error")
    plt.show()


def order_of_schemes_on_h():
    i_errors = []
    cn_errors = []
    hs = []
    for i in range(1, 15):
        m = 10 + int(np.sqrt(2) ** i)
        print(m, end=' : ')
        n = 5000
        hs.append((RIGHT - LEFT) / (m - 1))
        i_errors.append(implicit_scheme_error(m, n))
        print("Implicit ", end='')
        cn_errors.append(crank_nicolson_error(m, n))
        print("Crank-Nicolson ")

    plt.grid()
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(hs, i_errors, label="implicit")
    plt.plot(hs, cn_errors, label="crank nicolson")
    plt.legend()
    plt.xlabel("h")
    plt.ylabel("error")
    plt.show()


def order_of_schemes_on_tau():
    i_errors = []
    cn_errors = []
    hs = []
    for i in range(3, 8):
        n = 2 ** i
        print(n, end=' : ')
        m = 1000
        hs.append(T / (n - 1))
        i_errors.append(implicit_scheme_error(m, n))
        print("Implicit ", end='')
        cn_errors.append(crank_nicolson_error(m, n))
        print("Crank-Nicolson ")

    plt.grid()
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(hs, i_errors, label="implicit")
    plt.plot(hs, cn_errors, label="crank nicolson")
    plt.legend()
    plt.xlabel("$\\tau$")
    plt.ylabel("error")
    plt.show()


def time_search():
    ex_errors = []
    ex_times = []
    im_errors = []
    im_times = []
    cn_errors = []
    cn_times = []
    h_0 = 0.2
    for i in range(14):
        h = h_0 / (np.sqrt(2) ** i)
        m = int((RIGHT - LEFT) / h) + 1
        n = int(T / (h * h)) + 1
        print(i)
        if i < 9:
            start = time.time()
            ex_errors.append(explicit_scheme_error(m, 0))
            ex_times.append((time.time() - start) / 3)
            start = time.time()
            im_errors.append(implicit_scheme_error(m, n))
            im_times.append((time.time() - start) / 3)
        start = time.time()
        cn_errors.append(crank_nicolson_error(m, T*m))
        cn_times.append((time.time() - start) / 3)

    plt.grid()
    plt.plot(ex_errors, ex_times, label="Explicit")
    plt.plot(im_errors, im_times, label="Implicit")
    plt.plot(cn_errors, cn_times, label="Crank-Nicolson")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("error")
    plt.ylabel("time")
    plt.show()


explicit_scheme_plot_sols(30)
# x = np.linspace(LEFT, RIGHT, 100)
# t = np.linspace(0, T, 300)
# t, x = np.meshgrid(t, x)
# z = exact_func(x, t)
# plt.contourf(t, x, z, 26)
# plt.colorbar()
# plt.xlabel("t")
# plt.ylabel("x")
# plt.show()
