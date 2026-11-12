"""CEC 2022 single-objective bound-constrained benchmark suite.

The official shift, rotation, and shuffle data are stored in
``benchmarks/cec2022_data``. See README.md for the source and usage.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np


LOWER_BOUND = -100.0
UPPER_BOUND = 100.0
SUPPORTED_DIMENSIONS = (10, 20)
OPTIMAL_VALUES = {
    1: 300.0, 2: 400.0, 3: 600.0, 4: 800.0, 5: 900.0,
    6: 1800.0, 7: 2000.0, 8: 2200.0, 9: 2300.0,
    10: 2400.0, 11: 2600.0, 12: 2700.0,
}
FUNCTION_NAMES = {
    1: "Zakharov", 2: "Rosenbrock", 3: "Expanded Schaffer F7",
    4: "Non-continuous Rastrigin", 5: "Levy",
    6: "Hybrid Function 1", 7: "Hybrid Function 2",
    8: "Hybrid Function 3", 9: "Composition Function 1",
    10: "Composition Function 2", 11: "Composition Function 3",
    12: "Composition Function 4",
}

_DATA_DIR = Path(__file__).with_name("cec2022_data")
_PI = np.pi


@lru_cache(maxsize=None)
def _load_data(function_number, dimension):
    prefix = _DATA_DIR
    shift_data = np.loadtxt(prefix / f"shift_data_{function_number}.txt")
    if function_number <= 8:
        shift = np.asarray(shift_data, dtype=float).reshape(-1)[:dimension]
        rotation = np.loadtxt(prefix / f"M_{function_number}_D{dimension}.txt")
        rotations = np.asarray(rotation, dtype=float).reshape(1, dimension, dimension)
        shifts = shift.reshape(1, dimension)
    else:
        # Composition shift files store one row per optimum, padded to 100
        # columns. Keep the first D values from each row; flattening the file
        # would interleave components and corrupt their optimum locations.
        shift_rows = np.asarray(shift_data, dtype=float)
        if shift_rows.ndim == 1:
            shift_rows = shift_rows.reshape(1, -1)
        shifts = shift_rows[:10, :dimension]
        rotation = np.loadtxt(prefix / f"M_{function_number}_D{dimension}.txt")
        rotations = np.asarray(rotation, dtype=float).reshape(-1, dimension, dimension)[:10]

    shuffle = None
    if 6 <= function_number <= 8:
        shuffle = np.asarray(
            np.loadtxt(prefix / f"shuffle_data_{function_number}_D{dimension}.txt"),
            dtype=int,
        ).reshape(-1) - 1
    return shifts, rotations, shuffle


def _transform(x, shift, rotation, scale=1.0):
    """CEC shift-scale-rotate transform using the reference matrix layout."""
    return ((np.asarray(x, dtype=float) - shift) * scale) @ rotation.T


def _expanded_schaffer_f7(x):
    x = np.asarray(x, dtype=float)
    pairs = np.column_stack((x[:-1], x[1:]))
    radii = np.sum(pairs * pairs, axis=1)
    s = np.sqrt(radii)
    terms = np.sqrt(s) * (1.0 + np.sin(50.0 * np.power(s, 0.2)) ** 2)
    return float(np.sum(terms) ** 2 / (len(x) - 1) ** 2)


def _expanded_schaffer_f6(x):
    x = np.asarray(x, dtype=float)
    following = np.roll(x, -1)
    radius_squared = x * x + following * following
    numerator = np.sin(np.sqrt(radius_squared)) ** 2 - 0.5
    return float(np.sum(0.5 + numerator / (1.0 + 0.001 * radius_squared) ** 2))


def _zakharov(x):
    indices = np.arange(1, len(x) + 1, dtype=float)
    weighted_sum = 0.5 * np.dot(indices, x)
    return float(np.dot(x, x) + weighted_sum**2 + weighted_sum**4)


def _rosenbrock(x):
    x = np.asarray(x, dtype=float) + 1.0
    return float(np.sum(100.0 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1.0) ** 2))


def _rastrigin(x):
    return float(np.sum(x * x - 10.0 * np.cos(2.0 * _PI * x) + 10.0))


def _levy(x):
    w = 1.0 + np.asarray(x, dtype=float) / 4.0
    middle = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(_PI * w[:-1] + 1.0) ** 2))
    return float(np.sin(_PI * w[0]) ** 2 + middle + (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * _PI * w[-1]) ** 2))


def _bent_cigar(x):
    return float(x[0] ** 2 + 1.0e6 * np.dot(x[1:], x[1:]))


def _hgbat(x):
    z = np.asarray(x, dtype=float) - 1.0
    r2 = np.dot(z, z)
    total = np.sum(z)
    return float(np.sqrt(abs(r2 * r2 - total * total)) + (0.5 * r2 + total) / len(z) + 0.5)


def _elliptic(x):
    exponents = np.arange(len(x), dtype=float) / max(len(x) - 1, 1)
    return float(np.sum(np.power(1.0e6, exponents) * np.asarray(x) ** 2))


def _katsuura(x):
    x = np.asarray(x, dtype=float)
    j = np.arange(1, 33, dtype=float)
    powers = np.exp2(j)
    distances = np.abs(x[:, None] * powers - np.floor(x[:, None] * powers + 0.5)) / powers
    sums = np.sum(distances, axis=1)
    base = np.prod(np.power(1.0 + np.arange(1, len(x) + 1) * sums, 10.0 / len(x) ** 1.2))
    return float(10.0 / len(x) ** 2 * (base - 1.0))


def _happycat(x):
    z = np.asarray(x, dtype=float) - 1.0
    r2 = np.dot(z, z)
    return float(abs(r2 - len(z)) ** 0.25 + (0.5 * r2 + np.sum(z)) / len(z) + 0.5)


def _griewank(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * x) / 4000.0 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1.0)


def _griewank_rosenbrock(x):
    z = np.asarray(x, dtype=float) + 1.0
    nxt = np.roll(z, -1)
    t = 100.0 * (z * z - nxt) ** 2 + (z - 1.0) ** 2
    return float(np.sum(t * t / 4000.0 - np.cos(t) + 1.0))


def _schwefel(x, dimension):
    z = np.asarray(x, dtype=float) * 10.0 + 420.9687462275036
    mod = np.fmod(np.abs(z), 500.0)
    folded = np.where(z > 500.0, 500.0 - np.fmod(z, 500.0), np.where(z < -500.0, mod - 500.0, z))
    values = -folded * np.sin(np.sqrt(np.abs(folded)))
    values += np.where(z > 500.0, ((z - 500.0) / 100.0) ** 2 / dimension, 0.0)
    values += np.where(z < -500.0, ((z + 500.0) / 100.0) ** 2 / dimension, 0.0)
    return float(418.9828872724338 * dimension + np.sum(values))


def _ackley(x):
    x = np.asarray(x, dtype=float)
    return float(-20.0 * np.exp(-0.2 * np.sqrt(np.mean(x * x))) - np.exp(np.mean(np.cos(2.0 * _PI * x))) + 20.0 + np.e)


def _discus(x):
    return float(1.0e6 * x[0] ** 2 + np.dot(x[1:], x[1:]))


def _basic(function_id, x):
    x = np.asarray(x, dtype=float)
    scale = {
        2: 2.048 / 100.0, 4: 5.12 / 100.0, 7: 5.0 / 100.0,
        9: 5.0 / 100.0, 10: 5.0 / 100.0, 11: 5.0 / 100.0,
        15: 600.0 / 100.0,
    }.get(function_id, 1.0)
    x = x * scale
    functions = {
        2: _rosenbrock, 3: _expanded_schaffer_f6, 4: _rastrigin,
        6: _bent_cigar, 7: _hgbat, 8: _elliptic, 9: _katsuura,
        10: _happycat, 11: _griewank_rosenbrock, 13: _ackley,
        14: _discus, 15: _griewank, 16: _expanded_schaffer_f7,
    }
    if function_id == 1:
        return _zakharov(x)
    if function_id == 5:
        return _levy(x)
    if function_id == 12:
        return _schwefel(x, len(x))
    return functions[function_id](x)


def _component(function_id, x, shift, rotation, scale, rotate=True):
    z = _transform(x, shift, rotation, scale) if rotate else (np.asarray(x) - shift) * scale
    return _basic(function_id, z)


def _hybrid(x, shifts, rotations, shuffle, components):
    x = _transform(x, shifts[0], rotations[0])
    permuted = x[shuffle]
    dimension = len(x)
    sizes = [int(np.ceil(percent * dimension)) for percent, _ in components[:-1]]
    sizes.append(dimension - sum(sizes))
    total, start = 0.0, 0
    for size, (_, function_id) in zip(sizes, components):
        total += _basic(function_id, permuted[start:start + size])
        start += size
    return total


_HYBRID_COMPONENTS = {
    6: ((0.4, 6), (0.4, 7), (0.2, 4)),
    7: ((0.1, 7), (0.2, 9), (0.2, 13), (0.2, 4), (0.1, 12), (0.2, 16)),
    8: ((0.3, 9), (0.2, 10), (0.2, 11), (0.1, 12), (0.2, 13)),
}

_COMPOSITION_COMPONENTS = {
    9: ((2, 1.0e4, 1.0e4), (8, 1.0e4, 1.0e10), (6, 1.0e4, 1.0e30), (14, 1.0e4, 1.0e10), (8, 1.0e4, 1.0e10)),
    10: ((12, 1.0, 1.0), (4, 1.0, 1.0), (7, 1.0, 1.0)),
    11: ((3, 1.0e4, 2.0e7), (12, 1.0, 1.0), (15, 10.0, 1.0), (2, 1.0, 1.0), (4, 10.0, 1.0)),
    12: ((7, 1.0e4, 1.0e3), (4, 1.0e4, 1.0e3), (12, 1.0e4, 4.0e3), (6, 1.0e4, 1.0e30), (8, 1.0e4, 1.0e10), (3, 1.0e4, 2.0e7)),
}
_COMPOSITION_SIGMAS = {
    9: (10, 20, 30, 40, 50), 10: (20, 10, 10),
    11: (20, 20, 30, 30, 20), 12: (10, 20, 30, 40, 50, 60),
}
_COMPOSITION_BIASES = {
    9: (0, 200, 300, 100, 400), 10: (0, 200, 100),
    11: (0, 200, 300, 400, 200), 12: (0, 300, 500, 100, 400, 200),
}


def _composition(function_number, x, shifts, rotations):
    components = _COMPOSITION_COMPONENTS[function_number]
    sigmas = np.asarray(_COMPOSITION_SIGMAS[function_number], dtype=float)
    biases = np.asarray(_COMPOSITION_BIASES[function_number], dtype=float)
    values = []
    for index, (function_id, multiplier, denominator) in enumerate(components):
        rotate = not (function_number == 9 and index == 4) and not (function_number == 10 and index == 0)
        raw = _component(function_id, x, shifts[index], rotations[index], 1.0, rotate)
        values.append(multiplier * raw / denominator)

    distances = np.sum((np.asarray(x)[None, :] - shifts[:len(components)]) ** 2, axis=1)
    weights = np.empty_like(distances)
    nonzero = distances > 0.0
    weights[nonzero] = np.exp(-distances[nonzero] / (2.0 * len(x) * sigmas[nonzero] ** 2)) / np.sqrt(distances[nonzero])
    weights[~nonzero] = np.inf
    if np.any(np.isinf(weights)):
        weights = np.isinf(weights).astype(float)
        weights /= np.sum(weights)
    else:
        weight_sum = np.sum(weights)
        weights = weights / weight_sum if weight_sum > 0.0 else np.full_like(weights, 1.0 / len(weights))
    return float(np.dot(weights, np.asarray(values) + biases))


def cec2022(x, function_number):
    """Evaluate one CEC2022 function for a 10- or 20-dimensional vector."""
    if function_number not in OPTIMAL_VALUES:
        raise ValueError("function_number must be between 1 and 12")
    candidate = np.asarray(x, dtype=float)
    if candidate.ndim != 1 or candidate.size not in SUPPORTED_DIMENSIONS:
        raise ValueError("CEC2022 benchmark vectors must have dimension 10 or 20")
    if not np.all(np.isfinite(candidate)):
        raise ValueError("x must contain only finite values")
    if np.any(candidate < LOWER_BOUND) or np.any(candidate > UPPER_BOUND):
        raise ValueError("x must lie within [-100, 100]")

    dimension = candidate.size
    shifts, rotations, shuffle = _load_data(function_number, dimension)
    if function_number == 1:
        result = _component(1, candidate, shifts[0], rotations[0], 1.0)
    elif function_number == 2:
        result = _component(2, candidate, shifts[0], rotations[0], 1.0)
    elif function_number == 3:
        result = _component(16, candidate, shifts[0], rotations[0], 0.5 / 100.0)
    elif function_number == 4:
        centered = candidate - shifts[0]
        stepped = np.where(np.abs(centered) > 0.5, np.floor(2.0 * centered + 0.5) / 2.0, centered)
        result = _basic(4, stepped @ rotations[0].T)
    elif function_number == 5:
        result = _component(5, candidate, shifts[0], rotations[0], 5.12 / 100.0)
    elif function_number <= 8:
        result = _hybrid(candidate, shifts, rotations, shuffle, _HYBRID_COMPONENTS[function_number])
    else:
        result = _composition(function_number, candidate, shifts, rotations)
    return float(result + OPTIMAL_VALUES[function_number])


def make_cec2022(function_number):
    """Return a one-argument objective callable compatible with optimizers."""
    if function_number not in OPTIMAL_VALUES:
        raise ValueError("function_number must be between 1 and 12")
    return lambda x: cec2022(x, function_number)


CEC2022_BENCHMARKS = tuple(
    (number, FUNCTION_NAMES[number], OPTIMAL_VALUES[number], LOWER_BOUND, UPPER_BOUND)
    for number in range(1, 13)
)
