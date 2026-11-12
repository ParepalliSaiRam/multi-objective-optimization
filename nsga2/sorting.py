import numpy as np


def dominates(a, b):
    """
    Return True if solution a dominates solution b.

    Assumes all objectives are minimization objectives.

    a dominates b if:
        1. a is no worse than b in every objective
        2. a is strictly better in at least one objective
    """

    if a.constraint_violation == 0.0 and b.constraint_violation > 0.0:
        return True
    if a.constraint_violation > 0.0 and b.constraint_violation == 0.0:
        return False
    if a.constraint_violation > 0.0 and b.constraint_violation > 0.0:
        return a.constraint_violation < b.constraint_violation

    no_worse = all(
        x <= y
        for x, y in zip(a.objectives, b.objectives)
    )

    strictly_better = any(
        x < y
        for x, y in zip(a.objectives, b.objectives)
    )

    return no_worse and strictly_better


def fast_non_dominated_sort(population):
    """
    Perform fast non-dominated sorting.

    Returns:
        fronts = [F1, F2, F3, ...]
    """

    if not population:
        return []

    objectives = np.asarray([individual.objectives for individual in population], dtype=float)
    violations = np.asarray([individual.constraint_violation for individual in population], dtype=float)
    no_worse = np.all(objectives[:, None, :] <= objectives[None, :, :], axis=2)
    strictly_better = np.any(objectives[:, None, :] < objectives[None, :, :], axis=2)
    objective_dominance = no_worse & strictly_better
    feasible = violations == 0.0
    dominance = (
        (feasible[:, None] & ~feasible[None, :])
        | (~feasible[:, None] & ~feasible[None, :] & (violations[:, None] < violations[None, :]))
        | (feasible[:, None] & feasible[None, :] & objective_dominance)
    )
    np.fill_diagonal(dominance, False)

    fronts = [[]]
    for index, individual in enumerate(population):
        dominated_indices = np.flatnonzero(dominance[index])
        individual.dominated_solutions = [population[j] for j in dominated_indices]
        individual.domination_count = int(np.count_nonzero(dominance[:, index]))
        individual.rank = None
        if individual.domination_count == 0:
            individual.rank = 1
            fronts[0].append(individual)

    # Generate subsequent fronts
    i = 0

    while fronts[i]:

        next_front = []

        for p in fronts[i]:

            for q in p.dominated_solutions:

                q.domination_count -= 1

                if q.domination_count == 0:
                    q.rank = i + 2
                    next_front.append(q)

        i += 1

        if next_front:
            fronts.append(next_front)
        else:
            break

    return fronts


def calculate_crowding_distance(front):
    """
    Calculate crowding distance for all individuals
    in a single Pareto front.
    """

    if not front:
        return

    # Reset
    for individual in front:
        individual.crowding_distance = 0.0

    # A front with 1 or 2 individuals:
    # all boundary individuals get infinity.
    if len(front) <= 2:
        for individual in front:
            individual.crowding_distance = float("inf")
        return

    number_of_objectives = len(front[0].objectives)

    for objective_index in range(number_of_objectives):

        # Sort by current objective
        front.sort(
            key=lambda individual:
                individual.objectives[objective_index]
        )

        # Boundary solutions
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")

        objective_min = front[0].objectives[objective_index]
        objective_max = front[-1].objectives[objective_index]

        # Avoid division by zero
        if objective_max == objective_min:
            continue

        for i in range(1, len(front) - 1):

            previous_value = (
                front[i - 1].objectives[objective_index]
            )

            next_value = (
                front[i + 1].objectives[objective_index]
            )

            distance = (
                next_value - previous_value
            ) / (objective_max - objective_min)

            front[i].crowding_distance += distance
