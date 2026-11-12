"""Differential Evolution using the DE/rand/1/bin strategy."""

import numpy as np


class DifferentialEvolution:
    """CEC protocol compatible Differential Evolution optimizer.

    The default settings are shared across every function and dimension:
    population size 100, differential weight F=0.5, and crossover rate CR=0.9.
    """

    def __init__(self, population_size=100, differential_weight=0.5, crossover_rate=0.9):
        if population_size < 4:
            raise ValueError("population_size must be at least 4 for DE/rand/1")
        if not 0.0 < differential_weight <= 2.0:
            raise ValueError("differential_weight must be in (0, 2]")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        self.population_size = int(population_size)
        self.differential_weight = float(differential_weight)
        self.crossover_rate = float(crossover_rate)

    def __call__(self, objective, lower_bounds, upper_bounds, max_fes, seed):
        """Optimize ``objective`` within the supplied bounds and FE budget."""
        lower = np.asarray(lower_bounds, dtype=float)
        upper = np.asarray(upper_bounds, dtype=float)
        if lower.ndim != 1 or upper.shape != lower.shape or lower.size == 0:
            raise ValueError("lower_bounds and upper_bounds must be non-empty vectors of equal size")
        if np.any(lower >= upper):
            raise ValueError("each lower bound must be smaller than its upper bound")
        if max_fes < 4:
            raise ValueError("max_fes must be at least 4 for DE/rand/1")

        rng = np.random.default_rng(int(seed))
        dimension = lower.size
        population_size = min(self.population_size, int(max_fes))
        population = rng.uniform(lower, upper, size=(population_size, dimension))
        fitness = np.empty(population_size, dtype=float)
        evaluations = 0
        available_indices = np.arange(population_size - 1)

        for index in range(population_size):
            fitness[index] = objective(population[index])
            evaluations += 1

        while evaluations < max_fes:
            for target_index in range(population_size):
                if evaluations >= max_fes:
                    break

                candidates = rng.choice(available_indices, size=3, replace=False)
                candidates += candidates >= target_index
                a, b, c = candidates
                donor = (
                    population[a]
                    + self.differential_weight * (population[b] - population[c])
                )
                donor = np.clip(donor, lower, upper)

                crossover = rng.random(dimension) < self.crossover_rate
                crossover[rng.integers(dimension)] = True
                trial = np.where(crossover, donor, population[target_index])
                trial = np.clip(trial, lower, upper)
                trial_fitness = objective(trial)
                evaluations += 1

                if trial_fitness <= fitness[target_index]:
                    population[target_index] = trial
                    fitness[target_index] = trial_fitness

        best_index = int(np.argmin(fitness))
        return {
            "x": population[best_index].copy(),
            "value": float(fitness[best_index]),
            "evaluations": evaluations,
        }


DEFAULT_DE_PARAMETERS = {
    "strategy": "DE/rand/1/bin",
    "population_size": 100,
    "parameter_ranges": {
        "population_size": "integer in [4, MaxFES]",
        "differential_weight": "(0, 2]",
        "crossover_rate": "[0, 1]",
    },
    "differential_weight": 0.5,
    "crossover_rate": 0.9,
    "adjustment_guideline": "change population_size, F, or CR only as a shared setting after general tuning; do not tune by function or dimension",
    "parameter_tuning_cost_fes": 0,
    "initialization": "uniform within bounds",
    "boundary_repair": "clip to bounds",
    "selection": "greedy one-to-one",
    "encoding": "real-valued vectors; no separate encoding",
    "random_generator": "NumPy default_rng (PCG64), seeded by the official CEC numeric seed",
    "tuning": "none; one shared configuration for all functions and dimensions",
}
