import random


def crowded_comparison(a, b):
    """
    Return True if a is preferred over b
    according to NSGA-II's crowded comparison.

    Lower rank is better.
    If ranks are equal, higher crowding distance is better.
    """

    if a.rank < b.rank:
        return True

    if a.rank > b.rank:
        return False

    return a.crowding_distance > b.crowding_distance


def tournament_selection(population):
    """
    Perform binary tournament selection.
    """

    a = random.choice(population)
    b = random.choice(population)

    if crowded_comparison(a, b):
        return a

    return b