import math

def hypervolume_2d(points, reference_point):
    """
    Calculate the hypervolume for a 2-objective minimization problem.

    Parameters
    ----------
    points : list of [f1, f2]
        Obtained non-dominated solutions.

    reference_point : [r1, r2]
        Reference point. It must be worse than the relevant
        objective values.

    Returns
    -------
    float
        Hypervolume.
    """

    if not points:
        return 0.0

    # Remove points that are outside the reference point.
    filtered_points = [
        point
        for point in points
        if point[0] <= reference_point[0]
        and point[1] <= reference_point[1]
    ]

    if not filtered_points:
        return 0.0

    # Sort by first objective in ascending order.
    sorted_points = sorted(
        filtered_points,
        key=lambda point: point[0]
    )

    hypervolume = 0.0

    previous_f1 = reference_point[0]

    # For minimization:
    # move from the worst f1 toward the best f1.
    for f1, f2 in reversed(sorted_points):

        width = previous_f1 - f1
        height = reference_point[1] - f2

        if width > 0 and height > 0:
            hypervolume += width * height

        previous_f1 = f1

    return hypervolume

def igd(obtained_points, true_front):
    """
    Calculate Inverted Generational Distance (IGD).

    Parameters
    ----------
    obtained_points : list of [f1, f2]
        Pareto front obtained by the algorithm.

    true_front : list of [f1, f2]
        Reference/true Pareto front.

    Returns
    -------
    float
        IGD value. Lower is better.
    """

    if not true_front:
        return float("inf")

    if not obtained_points:
        return float("inf")

    total_distance = 0.0

    for reference_point in true_front:

        minimum_distance = float("inf")

        for obtained_point in obtained_points:

            distance = math.sqrt(
                sum(
                    (
                        reference_point[i]
                        - obtained_point[i]
                    ) ** 2
                    for i in range(len(reference_point))
                )
            )

            minimum_distance = min(
                minimum_distance,
                distance
            )

        total_distance += minimum_distance

    return total_distance / len(true_front)

def gd(obtained_points, true_front):

    if not obtained_points:
        return float("inf")

    if not true_front:
        return float("inf")

    total_squared_distance = 0.0

    for obtained_point in obtained_points:

        minimum_distance = float("inf")

        for reference_point in true_front:

            distance = math.sqrt(
                sum(
                    (obtained_point[i] - reference_point[i]) ** 2
                    for i in range(len(obtained_point))
                )
            )

            minimum_distance = min(
                minimum_distance,
                distance
            )

        total_squared_distance += (
            minimum_distance ** 2
        )

    return math.sqrt(
        total_squared_distance
        / len(obtained_points)
    )

def spacing(obtained_points):

    if len(obtained_points) < 2:
        return 0.0

    nearest_distances = []

    for i in range(len(obtained_points)):

        minimum_distance = float("inf")

        for j in range(len(obtained_points)):

            if i == j:
                continue

            distance = sum(
                abs(
                    obtained_points[i][k]
                    - obtained_points[j][k]
                )
                for k in range(
                    len(obtained_points[i])
                )
            )

            minimum_distance = min(
                minimum_distance,
                distance
            )

        nearest_distances.append(
            minimum_distance
        )

    mean_distance = (
        sum(nearest_distances)
        / len(nearest_distances)
    )

    squared_difference = sum(
        (distance - mean_distance) ** 2
        for distance in nearest_distances
    )

    return math.sqrt(
        squared_difference
        / (len(nearest_distances) - 1)
    )