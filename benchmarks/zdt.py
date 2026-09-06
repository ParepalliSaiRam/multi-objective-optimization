import math
import numpy as np


# ============================================================
# ZDT1
# ============================================================

def zdt1(x):

    f1 = x[0]

    g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)

    f2 = g * (
        1.0 - math.sqrt(f1 / g)
    )

    return [f1, f2]


def zdt1_bounds(number_of_variables):

    lower_bounds = [0.0] * number_of_variables
    upper_bounds = [1.0] * number_of_variables

    return lower_bounds, upper_bounds


def zdt1_pareto_front(number_of_points=300):

    f1 = np.linspace(0.0, 1.0, number_of_points)

    f2 = 1.0 - np.sqrt(f1)

    return [(f1, f2)]


# ============================================================
# ZDT2
# ============================================================

def zdt2(x):

    f1 = x[0]

    g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)

    f2 = g * (
        1.0 - (f1 / g) ** 2
    )

    return [f1, f2]


def zdt2_bounds(number_of_variables):

    lower_bounds = [0.0] * number_of_variables
    upper_bounds = [1.0] * number_of_variables

    return lower_bounds, upper_bounds


def zdt2_pareto_front(number_of_points=300):

    f1 = np.linspace(0.0, 1.0, number_of_points)

    f2 = 1.0 - f1 ** 2

    return [(f1, f2)]


# ============================================================
# ZDT3
# ============================================================

def zdt3(x):

    f1 = x[0]

    g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)

    f2 = g * (
        1.0
        - math.sqrt(f1 / g)
        - (f1 / g) * math.sin(10.0 * math.pi * f1)
    )

    return [f1, f2]


def zdt3_bounds(number_of_variables):

    lower_bounds = [0.0] * number_of_variables
    upper_bounds = [1.0] * number_of_variables

    return lower_bounds, upper_bounds

def zdt3_pareto_front(number_of_points=1000):

    intervals = [
        (0.0, 0.0830015349),
        (0.1822287280, 0.2577623634),
        (0.4093136748, 0.4538821041),
        (0.6183967944, 0.6525117038),
        (0.8233317983, 0.8518328654)
    ]

    fronts = []

    points_per_interval = number_of_points // len(intervals)

    for lower, upper in intervals:

        f1 = np.linspace(
            lower,
            upper,
            points_per_interval
        )

        f2 = (
            1.0
            - np.sqrt(f1)
            - f1 * np.sin(10.0 * np.pi * f1)
        )

        fronts.append((f1, f2))

    return fronts

# ============================================================
# ZDT4
# ============================================================

def zdt4(x):

    f1 = x[0]

    g = (
        1.0
        + 10.0 * (len(x) - 1)
        + sum(
            xi ** 2
            - 10.0 * math.cos(4.0 * math.pi * xi)
            for xi in x[1:]
        )
    )

    f2 = g * (
        1.0 - math.sqrt(f1 / g)
    )

    return [f1, f2]


def zdt4_bounds(number_of_variables):

    lower_bounds = [0.0] + [-5.0] * (number_of_variables - 1)
    upper_bounds = [1.0] + [5.0] * (number_of_variables - 1)

    return lower_bounds, upper_bounds


def zdt4_pareto_front(number_of_points=300):

    f1 = np.linspace(0.0, 1.0, number_of_points)

    f2 = 1.0 - np.sqrt(f1)

    return [(f1, f2)]


# ============================================================
# ZDT6
# ============================================================

def zdt6(x):

    f1 = (
        1.0
        - math.exp(-4.0 * x[0])
        * math.sin(6.0 * math.pi * x[0]) ** 6
    )

    mean_rest = sum(x[1:]) / (len(x) - 1)

    g = 1.0 + 9.0 * mean_rest ** 0.25

    f2 = g * (
        1.0 - (f1 / g) ** 2
    )

    return [f1, f2]


def zdt6_bounds(number_of_variables):

    lower_bounds = [0.0] * number_of_variables
    upper_bounds = [1.0] * number_of_variables

    return lower_bounds, upper_bounds


def zdt6_pareto_front(number_of_points=1000):

    f1_min = 0.2807753258

    f1 = np.linspace(
        f1_min,
        1.0,
        number_of_points
    )

    f2 = 1.0 - f1 ** 2

    return [(f1, f2)]