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