"""Classical benchmark problems appearing in Benchmark_Comparison.pdf."""

import math
from functools import lru_cache

import numpy as np


def binh1(x):
    a, b = x
    return [a * a + b * b, (a - 5) ** 2 + (b - 5) ** 2]


def binh3(x):
    a, b = x
    return [a - 1e6, b - 2e-6, a * b - 2.0]


def deb(x):
    a = x[0]
    return [a, 1.0 - a * a]


def deb2(x):
    a, b = x
    g = 1 + 10 * b
    ratio = a / g
    return [a, g * (1 - ratio ** 2 - ratio * math.sin(8 * math.pi * a))]


def fonseca(x):
    a, b = x
    return [1 - math.exp(-((a - 1) ** 2 + (b + 1) ** 2)), 1 - math.exp(-((a + 1) ** 2 + (b - 1) ** 2))]


def fonseca2(x):
    shift = 1 / math.sqrt(len(x))
    return [
        1 - math.exp(-sum((value - shift) ** 2 for value in x)),
        1 - math.exp(-sum((value + shift) ** 2 for value in x)),
    ]


def kursawe(x):
    f1 = sum(-10 * math.exp(-0.2 * math.sqrt(x[i] ** 2 + x[i + 1] ** 2)) for i in range(len(x) - 1))
    f2 = sum(abs(value) ** 0.8 + 5 * math.sin(value ** 3) for value in x)
    return [f1, f2]


def laumanns(x):
    a, b = x
    return [a * a + b * b, (a + 2) ** 2 + b * b]


def murata(x):
    a, b = x
    return [2 * math.sqrt(a), a * (1 - b) + 5]


def quagliarella(x):
    value = x[0]
    first = value * value + 10 * (1 - math.cos(2 * math.pi * value))
    shifted = value - 1.5
    second = shifted * shifted + 10 * (1 - math.cos(2 * math.pi * shifted))
    return [math.sqrt(max(0.0, first)), math.sqrt(max(0.0, second))]


def rendon(x):
    a, b = x
    radius = a * a + b * b
    return [1 / (radius + 1), a * a + 3 * b * b + 1]


def rendon2(x):
    a, b = x
    return [a + b + 1, a * a + 2 * b - 1]


def schaffer(x):
    value = x[0]
    return [value * value, (value - 2) ** 2]


def schaffer2(x):
    value = x[0]
    if value <= 1:
        first = -value
    elif value <= 3:
        first = value - 2
    elif value <= 4:
        first = 4 - value
    else:
        first = value - 4
    return [first, (value - 5) ** 2]


def srinivas(x):
    a, b = x
    return [(a - 2) ** 2 + (b - 1) ** 2 + 2, 9 * a - (b - 1) ** 2]


def tanaka(x):
    return [x[0], x[1]]


def viennet(x):
    a, b = x
    radius = a * a + b * b
    return [
        0.5 * radius + math.sin(radius),
        (3 * a - 2 * b + 4) ** 2 / 8 + (a - b + 1) ** 2 / 27 + 15,
        1 / (radius + 1) - 1.1 * math.exp(-radius),
    ]


def viennet2(x):
    a, b = x
    return [
        (a - 2) ** 2 / 2 + (b + 1) ** 2 / 13 + 3,
        (a + b - 3) ** 2 / 36 + (-a + b + 2) ** 2 / 8 - 17,
        (a + 2 * b - 1) ** 2 / 175 + (2 * b - a) ** 2 / 17 - 13,
    ]


def srinivas_violation(x):
    a, b = x
    return max(0.0, a * a + b * b - 225) + max(0.0, a - 3 * b + 10)


def tanaka_violation(x):
    a, b = x
    theta = math.atan2(a, b)
    c1 = a * a + b * b - 1 - 0.1 * math.cos(16 * theta)
    c2 = (a - 0.5) ** 2 + (b - 0.5) ** 2 - 0.5
    return max(0.0, -c1) + max(0.0, c2)


def bounds(name, number_of_variables=None):
    name = name.lower().replace("(", "").replace(")", "")
    ranges = {
        "binh1": (-5, 10, 2), "binh3": (1e-6, 1e6, 2),
        "deb": (0, 1, 2), "deb2": (0, 1, 2), "fonseca": (-4, 4, 2),
        "fonseca2": (-4, 4, 3), "kursawe": (-5, 5, 3),
        "laumanns": (-50, 50, 2), "murata": (1, 4, 2),
        "quagliarella": (-5.12, 5.12, 1), "rendon": (-3, 3, 2),
        "rendon2": (-3, 3, 2), "schaffer": (-10, 10, 1),
        "schaffer2": (-5, 10, 1), "srinivas": (-20, 20, 2),
        "tanaka": (1e-5, math.pi, 2), "viennet": (-3, 3, 2),
        "viennet2": (-4, 4, 2),
    }
    low, high, count = ranges[name]
    if name == "murata":
        return [1.0, 1.0], [4.0, 2.0]
    return [low] * count, [high] * count


def _nondominated(points):
    if len(points) == 0:
        return np.empty((0, 2))
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if points.shape[1] == 2:
        order = np.lexsort((points[:, 1], points[:, 0]))
        points = points[order]
        best_before = np.concatenate(([math.inf], np.minimum.accumulate(points[:-1, 1])))
        return points[points[:, 1] < best_before]
    # Sweep objective 1 in ascending order. A Fenwick tree stores the best
    # (lowest) objective 3 seen for each prefix of objective 2, allowing a
    # dominance query in O(log n) rather than comparing every pair.
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    sorted_points = points[order]
    y_values = np.unique(sorted_points[:, 1])
    y_indices = np.searchsorted(y_values, sorted_points[:, 1]) + 1
    tree = np.full(len(y_values) + 1, math.inf)
    keep = np.zeros(len(sorted_points), dtype=bool)
    for index, (point, y_index) in enumerate(zip(sorted_points, y_indices)):
        best_z = math.inf
        cursor = y_index
        while cursor > 0:
            best_z = min(best_z, tree[cursor])
            cursor -= cursor & -cursor
        if best_z > point[2]:
            keep[index] = True
            cursor = y_index
            while cursor < len(tree):
                tree[cursor] = min(tree[cursor], point[2])
                cursor += cursor & -cursor
    return sorted_points[keep]


@lru_cache(maxsize=None)
def reference_front(name, number_of_points=1000):
    """Generate an objective-space reference sample for a classic problem."""
    name = name.lower().replace("(", "").replace(")", "")
    if name == "binh3":
        return np.asarray([binh3([1e-6, 1e-6])])
    if name == "binh1":
        values = np.linspace(0, 5, number_of_points)
        return np.asarray([binh1([value, value]) for value in values])
    if name == "deb2":
        values = np.linspace(0, 1, max(5000, number_of_points))
        return _nondominated([deb2([value, 0.0]) for value in values])
    if name == "laumanns":
        values = np.linspace(-2, 0, number_of_points)
        return np.asarray([laumanns([value, 0]) for value in values])
    if name == "murata":
        values = np.linspace(1, 4, number_of_points)
        return np.asarray([murata([value, 2]) for value in values])
    if name == "rendon2":
        values = np.linspace(-3, 0, number_of_points)
        return np.asarray([rendon2([value, -3]) for value in values])
    if name == "schaffer":
        values = np.linspace(0, 2, number_of_points)
        return np.asarray([schaffer([value]) for value in values])
    if name == "fonseca":
        values = np.linspace(-1, 1, number_of_points)
        return np.asarray([fonseca([value, -value]) for value in values])
    if name == "fonseca2":
        values = np.linspace(-1 / math.sqrt(3), 1 / math.sqrt(3), number_of_points)
        return np.asarray([fonseca2([value] * 3) for value in values])
    if name == "tanaka":
        x_values = np.linspace(0.0, math.pi, 2400)
        y_values = np.linspace(0.0, math.pi, 4000)
        candidates = []
        for a in x_values:
            c1 = a * a + y_values * y_values - 1 - 0.1 * np.cos(16 * np.arctan2(a, y_values))
            c2 = (a - 0.5) ** 2 + (y_values - 0.5) ** 2 - 0.5
            feasible_indices = np.flatnonzero((c1 >= 0.0) & (c2 <= 0.0))
            if feasible_indices.size:
                candidates.append([a, y_values[feasible_indices[0]]])
        return _nondominated(candidates)
    if name == "srinivas":
        values = np.linspace(-20, 20, 1201)
        a, b = np.meshgrid(values, values, indexing="ij")
        feasible = ((a * a + b * b <= 225.0) & (a - 3 * b + 10.0 <= 0.0))
        first = (a - 2.0) ** 2 + (b - 1.0) ** 2 + 2.0
        second = 9.0 * a - (b - 1.0) ** 2
        return _nondominated(np.column_stack((first[feasible], second[feasible])))
    if name in {"viennet", "viennet2"}:
        low, high = (-3.0, 3.0) if name == "viennet" else (-4.0, 4.0)
        values = np.linspace(low, high, 401)
        func = viennet if name == "viennet" else viennet2
        a, b = np.meshgrid(values, values, indexing="ij")
        radius = a * a + b * b
        if name == "viennet":
            candidates = np.column_stack((
                (0.5 * radius + np.sin(radius)).ravel(),
                ((3 * a - 2 * b + 4) ** 2 / 8 + (a - b + 1) ** 2 / 27 + 15).ravel(),
                (1 / (radius + 1) - 1.1 * np.exp(-radius)).ravel(),
            ))
        else:
            candidates = np.column_stack((
                ((a - 2) ** 2 / 2 + (b + 1) ** 2 / 13 + 3).ravel(),
                ((a + b - 3) ** 2 / 36 + (-a + b + 2) ** 2 / 8 - 17).ravel(),
                ((a + 2 * b - 1) ** 2 / 175 + (2 * b - a) ** 2 / 17 - 13).ravel(),
            ))
        return _nondominated(candidates)
    low, high, dimensions = {
        "binh1": (-5, 10, 2), "deb": (0, 1, 1), "deb2": (0, 1, 2),
        "rendon": (-3, 3, 2),
        "kursawe": (-5, 5, 3), "quagliarella": (-5.12, 5.12, 1),
        "schaffer2": (-5, 10, 1),
    }[name]
    func = globals()[name]
    if dimensions == 1:
        sample_count = max(number_of_points, 20000) if name == "quagliarella" else number_of_points
        samples = np.linspace(low, high, sample_count)
        candidates = [func([value]) for value in samples]
    else:
        rng = np.random.default_rng(381)
        sample_count = max(number_of_points, 500000) if name == "kursawe" else max(number_of_points, 20000)
        samples = rng.uniform(low, high, size=(sample_count, dimensions))
        if name == "kursawe" and dimensions == 3:
            candidates = np.column_stack((
                -10 * np.exp(-0.2 * np.sqrt(samples[:, 0] ** 2 + samples[:, 1] ** 2))
                -10 * np.exp(-0.2 * np.sqrt(samples[:, 1] ** 2 + samples[:, 2] ** 2)),
                np.sum(np.abs(samples) ** 0.8 + 5 * np.sin(samples ** 3), axis=1),
            ))
        else:
            candidates = [func(row.tolist()) for row in samples]
    return _nondominated(candidates)


CLASSIC_BENCHMARKS = [
    ("Binh1", binh1, "binh1", None), ("Binh3", binh3, "binh3", None),
    ("Deb", deb, "deb", None), ("Deb2", deb2, "deb2", None),
    ("Fonseca", fonseca, "fonseca", None), ("Fonseca2", fonseca2, "fonseca2", None),
    ("Kursawe", kursawe, "kursawe", None), ("Laumanns", laumanns, "laumanns", None),
    ("Murata", murata, "murata", None), ("Quagliarella", quagliarella, "quagliarella", None),
    ("Rendon", rendon, "rendon", None), ("Rendon2", rendon2, "rendon2", None),
    ("Schaffer", schaffer, "schaffer", None), ("Schaffer2", schaffer2, "schaffer2", None),
    ("Srinivas", srinivas, "srinivas", srinivas_violation), ("Tanaka", tanaka, "tanaka", tanaka_violation),
    ("Viennet", viennet, "viennet", None), ("Viennet2", viennet2, "viennet2", None),
]
