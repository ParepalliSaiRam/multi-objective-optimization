import os

import matplotlib.pyplot as plt
import numpy as np


def _front_points(front):
    if hasattr(front, "ndim") and front.ndim == 2:
        return front.tolist()
    points = []
    for f1, f2 in front:
        points.extend([[float(x), float(y)] for x, y in zip(f1, f2)])
    return points


def _nondominated(points):
    """Return the nondominated union of objective vectors from all runs."""
    if not points:
        return []
    values = np.unique(np.asarray(points, dtype=float), axis=0)
    keep = np.ones(len(values), dtype=bool)
    # Objective sets here are small (at most a few hundred points per run).
    for i, point in enumerate(values):
        if np.any(np.all(values <= point, axis=1) & np.any(values < point, axis=1)):
            keep[i] = False
    return values[keep].tolist()


def save_pareto_front_plot(
    benchmark_name,
    true_front,
    obtained_fronts,
    output_directory="results/plots",
):
    """Save obtained and reference fronts side by side, matching the PDF layout."""
    os.makedirs(output_directory, exist_ok=True)
    reference = _front_points(true_front)
    # Each run is nondominated on its own, but points from one run can be
    # dominated by points found in another. Plot the true union front, rather
    # than a cloud containing those cross-run dominated points.
    obtained = _nondominated([point for run in obtained_fronts for point in run])
    dimensions = len(reference[0]) if reference else (len(obtained[0]) if obtained else 2)
    if dimensions not in (2, 3):
        raise ValueError("Plots support two- or three-objective fronts")

    figure = plt.figure(figsize=(12, 5))
    axes = [figure.add_subplot(1, 2, i + 1, projection="3d" if dimensions == 3 else None) for i in range(2)]
    for axis, title, points, color in (
        (axes[0], "Obtained Front", obtained, "#1678b8"),
        (axes[1], "True Pareto Front", reference, "#d62728"),
    ):
        if points:
            columns = list(zip(*points))
            if dimensions == 3:
                axis.scatter(*columns, s=12, alpha=0.72, color=color, label=title)
                axis.set_zlabel("f3")
                axis.view_init(elev=25, azim=45)
            else:
                axis.scatter(columns[0], columns[1], s=12, alpha=0.72, color=color, label=title)
            axis.set_xlabel("f1")
            axis.set_ylabel("f2")
            axis.set_title(f"{benchmark_name} - {title}")
            axis.grid(True, alpha=0.25)
            axis.legend(loc="best", fontsize=8)

    figure.tight_layout()
    output_path = os.path.join(output_directory, f"{benchmark_name.lower()}_pareto_front.png")
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


def save_pareto_set_plot(
    benchmark_name,
    true_set,
    obtained_sets,
    output_directory="results/plots",
):
    """Save obtained and reference Pareto sets in decision-variable space."""
    os.makedirs(output_directory, exist_ok=True)
    reference = np.asarray(true_set, dtype=float)
    obtained = np.asarray(
        [point for run in obtained_sets for point in run], dtype=float
    )
    if reference.ndim != 2 or reference.shape[1] not in (2, 3):
        raise ValueError("Pareto-set plots support two or three decision variables")
    if obtained.size == 0:
        obtained = np.empty((0, reference.shape[1]))
    else:
        obtained = np.unique(obtained.reshape(-1, reference.shape[1]), axis=0)

    dimensions = reference.shape[1]
    figure = plt.figure(figsize=(12, 5))
    axes = [
        figure.add_subplot(1, 2, i + 1, projection="3d" if dimensions == 3 else None)
        for i in range(2)
    ]
    for axis, title, points, color in (
        (axes[0], "Obtained Pareto Set", obtained, "#1678b8"),
        (axes[1], "Reference Pareto Set", reference, "#d62728"),
    ):
        if len(points):
            columns = points.T
            if dimensions == 3:
                axis.scatter(*columns, s=10, alpha=.65, color=color, label=title)
                axis.set_zlabel("x3")
                axis.view_init(elev=25, azim=45)
            else:
                axis.scatter(columns[0], columns[1], s=10, alpha=.65, color=color, label=title)
            axis.set_xlabel("x1")
            axis.set_ylabel("x2")
            axis.set_title(f"{benchmark_name} - {title}")
            axis.grid(True, alpha=.25)
            axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    output_path = os.path.join(output_directory, f"{benchmark_name.lower()}_pareto_set.png")
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path
