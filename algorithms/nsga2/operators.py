import random


def sbx_crossover(
    parent1,
    parent2,
    lower_bounds,
    upper_bounds,
    probability=0.9,
    eta_c=20
):
    """
    Simulated Binary Crossover (SBX).

    parent1 and parent2 are Individual objects.

    Returns:
        child1_variables, child2_variables
    """

    child1 = parent1.variables.copy()
    child2 = parent2.variables.copy()

    # No crossover
    if random.random() > probability:
        return child1, child2

    for i in range(len(parent1.variables)):

        x1 = parent1.variables[i]
        x2 = parent2.variables[i]

        lower = lower_bounds[i]
        upper = upper_bounds[i]

        # If parents have identical values,
        # there is no meaningful crossover to perform.
        if abs(x1 - x2) <= 1e-14:
            continue

        # Ensure x1 < x2
        if x1 > x2:
            x1, x2 = x2, x1

        random_value = random.random()

        # ---- Calculate beta and alpha for child 1 ----

        beta = 1.0 + (
            2.0 * (x1 - lower) / (x2 - x1)
        )

        alpha = 2.0 - beta ** -(eta_c + 1.0)

        if random_value <= 1.0 / alpha:
            beta_q = (
                random_value * alpha
            ) ** (1.0 / (eta_c + 1.0))
        else:
            beta_q = (
                1.0 / (
                    2.0
                    - random_value * alpha
                )
            ) ** (1.0 / (eta_c + 1.0))

        child1_value = (
            0.5
            * (
                (x1 + x2)
                - beta_q * (x2 - x1)
            )
        )

        # ---- Calculate beta and alpha for child 2 ----

        beta = 1.0 + (
            2.0 * (upper - x2) / (x2 - x1)
        )

        alpha = 2.0 - beta ** -(eta_c + 1.0)

        if random_value <= 1.0 / alpha:
            beta_q = (
                random_value * alpha
            ) ** (1.0 / (eta_c + 1.0))
        else:
            beta_q = (
                1.0 / (
                    2.0
                    - random_value * alpha
                )
            ) ** (1.0 / (eta_c + 1.0))

        child2_value = (
            0.5
            * (
                (x1 + x2)
                + beta_q * (x2 - x1)
            )
        )

        # Ensure bounds
        child1_value = min(
            max(child1_value, lower),
            upper
        )

        child2_value = min(
            max(child2_value, lower),
            upper
        )

        # Randomly exchange children
        if random.random() <= 0.5:
            child1[i] = child2_value
            child2[i] = child1_value
        else:
            child1[i] = child1_value
            child2[i] = child2_value

    return child1, child2


def polynomial_mutation(variables, lower_bounds, upper_bounds,
                        probability=None, eta_m=20):

    mutated = variables.copy()

    number_of_variables = len(mutated)

    if probability is None:
        probability = 1.0 / number_of_variables

    for i in range(number_of_variables):

        if random.random() > probability:
            continue

        x = mutated[i]

        lower = lower_bounds[i]
        upper = upper_bounds[i]

        if upper == lower:
            continue

        delta1 = (x - lower) / (upper - lower)
        delta2 = (upper - x) / (upper - lower)

        random_value = random.random()

        mutation_power = 1.0 / (eta_m + 1.0)

        if random_value <= 0.5:

            xy = 1.0 - delta1

            val = (
                2.0 * random_value
                + (1.0 - 2.0 * random_value)
                * (xy ** (eta_m + 1.0))
            )

            delta_q = (
                val ** mutation_power
            ) - 1.0

        else:

            xy = 1.0 - delta2

            val = (
                2.0 * (1.0 - random_value)
                + 2.0
                * (random_value - 0.5)
                * (xy ** (eta_m + 1.0))
            )

            delta_q = (
                1.0 - val ** mutation_power
            )

        x = x + delta_q * (upper - lower)

        mutated[i] = min(max(x, lower), upper)

    return mutated