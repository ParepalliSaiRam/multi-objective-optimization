"""DTLZ1-DTLZ7 benchmark suite and three-objective reference fronts."""

import math

import numpy as np


def _validate(x, number_of_objectives):
    if len(x) <= number_of_objectives:
        raise ValueError("DTLZ requires more variables than objectives")


def _g1(values):
    return 100.0 * (len(values) + sum((v - 0.5) ** 2 - math.cos(20 * math.pi * (v - 0.5)) for v in values))


def _g2(values):
    return sum((v - 0.5) ** 2 for v in values)


def dtlz1(x, number_of_objectives=3):
    _validate(x, number_of_objectives)
    m = number_of_objectives
    g = _g1(x[m - 1:])
    f = [0.5 * (1 + g)] * m
    for i in range(m):
        f[i] *= math.prod(x[j] for j in range(m - i - 1))
        if i:
            f[i] *= 1 - x[m - i - 1]
    return f


def _dtlz_spherical(x, variant, number_of_objectives=3):
    _validate(x, number_of_objectives)
    m = number_of_objectives
    tail = x[m - 1:]
    if variant in (2, 3, 4):
        g = _g1(tail) if variant == 3 else _g2(tail)
        alpha = 100 if variant == 4 else 1
        angles = [v ** alpha * math.pi / 2 for v in x[:m - 1]]
        result = [1 + g] * m
        for i in range(m):
            result[i] *= math.prod(math.cos(a) for a in angles[:m - i - 1])
            if i:
                result[i] *= math.sin(angles[m - i - 1])
        return result
    g = sum((v - 0.5) ** 2 for v in tail) if variant == 5 else sum(v ** 0.1 for v in tail)
    theta = [x[0] * math.pi / 2]
    theta.extend(math.pi / (4 * (1 + g)) * (1 + 2 * g * x[i]) for i in range(1, m - 1))
    result = [1 + g] * m
    for i in range(m):
        result[i] *= math.prod(math.cos(a) for a in theta[:m - i - 1])
        if i:
            result[i] *= math.sin(theta[m - i - 1])
    return result


def dtlz2(x, number_of_objectives=3):
    return _dtlz_spherical(x, 2, number_of_objectives)


def dtlz3(x, number_of_objectives=3):
    return _dtlz_spherical(x, 3, number_of_objectives)


def dtlz4(x, number_of_objectives=3):
    return _dtlz_spherical(x, 4, number_of_objectives)


def dtlz5(x, number_of_objectives=3):
    return _dtlz_spherical(x, 5, number_of_objectives)


def dtlz6(x, number_of_objectives=3):
    return _dtlz_spherical(x, 6, number_of_objectives)


def dtlz7(x, number_of_objectives=3):
    _validate(x, number_of_objectives)
    m = number_of_objectives
    f = list(x[:m - 1])
    g = 1 + 9 * sum(x[m - 1:]) / (len(x) - m + 1)
    h = m - sum((value / (1 + g)) * (1 + math.sin(3 * math.pi * value)) for value in f)
    return f + [(1 + g) * h]


def dtlz_bounds(number_of_variables):
    return [0.0] * number_of_variables, [1.0] * number_of_variables


def _simplex_grid(divisions):
    for i in range(divisions + 1):
        for j in range(divisions + 1 - i):
            yield i / divisions, j / divisions, (divisions - i - j) / divisions


def dtlz_pareto_front(name, divisions=50, number_of_objectives=3):
    """Return sampled objective-space points approximating the true front."""
    if divisions < 1:
        raise ValueError("divisions must be positive")
    name = name.upper()
    m = number_of_objectives
    points = []
    if name == "DTLZ1":
        points = [tuple(0.5 * value for value in weights) for weights in _simplex_grid(divisions)]
    elif name in {"DTLZ2", "DTLZ3", "DTLZ4"}:
        points = []
        for weights in _simplex_grid(divisions):
            norm = math.sqrt(sum(value * value for value in weights))
            points.append(tuple(value / norm for value in weights) if norm else tuple([1.0] + [0.0] * (m - 1)))
    elif name in {"DTLZ5", "DTLZ6"}:
        for index in range(divisions + 1):
            angle = index / divisions * math.pi / 2
            points.append((math.cos(angle) / math.sqrt(2), math.cos(angle) / math.sqrt(2), math.sin(angle)))
    elif name == "DTLZ7":
        grid = np.linspace(0.0, 1.0, divisions + 1)
        for a in grid:
            for b in grid:
                h = 3 - sum(v * (1 + math.sin(3 * math.pi * v)) for v in (a, b))
                if h >= 0:
                    points.append((float(a), float(b), float(2 * h)))
        points.sort(key=lambda point: point[0])
        front = []
        for point in points:
            if not any(all(old[i] <= point[i] for i in range(3)) and any(old[i] < point[i] for i in range(3)) for old in front):
                front = [old for old in front if not (all(point[i] <= old[i] for i in range(3)) and any(point[i] < old[i] for i in range(3)))]
                front.append(point)
        points = front
    else:
        raise ValueError(f"Unknown DTLZ benchmark: {name}")
    return np.asarray(points, dtype=float)
