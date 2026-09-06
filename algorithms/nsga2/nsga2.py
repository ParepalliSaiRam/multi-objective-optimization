import random

from .individual import Individual
from .sorting import (
    fast_non_dominated_sort,
    calculate_crowding_distance
)
from .selection import tournament_selection
from .operators import (
    sbx_crossover,
    polynomial_mutation
)


class NSGA2:

    def __init__(
        self,
        population_size,
        generations,
        number_of_variables,
        lower_bounds,
        upper_bounds,
        objective_function,
        crossover_probability=0.9,
        mutation_probability=None,
        eta_c=15,
        eta_m=20
    ):

        self.population_size = population_size
        self.generations = generations
        self.number_of_variables = number_of_variables

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        self.objective_function = objective_function

        self.crossover_probability = crossover_probability

        if mutation_probability is None:
            mutation_probability = 1.0 / number_of_variables

        self.mutation_probability = mutation_probability

        self.eta_c = eta_c
        self.eta_m = eta_m

    def initialize_population(self):

        population = []

        for _ in range(self.population_size):

            variables = [
                random.uniform(
                    self.lower_bounds[i],
                    self.upper_bounds[i]
                )
                for i in range(self.number_of_variables)
            ]

            individual = Individual(variables)

            population.append(individual)

        return population

    def evaluate_population(self, population):

        for individual in population:
            individual.objectives = list(
                self.objective_function(
                    individual.variables
                )
            )

    def assign_rank_and_crowding(self, population):

        fronts = fast_non_dominated_sort(population)

        for front in fronts:
            calculate_crowding_distance(front)

        return fronts

    def create_offspring(self, population):

        offspring = []

        while len(offspring) < self.population_size:

            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            child1_variables, child2_variables = sbx_crossover(
                parent1,
                parent2,
                self.lower_bounds,
                self.upper_bounds,
                self.crossover_probability,
                self.eta_c
            )

            child1_variables = polynomial_mutation(
                child1_variables,
                self.lower_bounds,
                self.upper_bounds,
                self.mutation_probability,
                self.eta_m
            )

            child2_variables = polynomial_mutation(
                child2_variables,
                self.lower_bounds,
                self.upper_bounds,
                self.mutation_probability,
                self.eta_m
            )

            child1 = Individual(child1_variables)
            child2 = Individual(child2_variables)

            offspring.append(child1)

            if len(offspring) < self.population_size:
                offspring.append(child2)

        self.evaluate_population(offspring)

        return offspring

    def environmental_selection(self, combined_population):

        fronts = fast_non_dominated_sort(combined_population)

        new_population = []

        i = 0

        while (
            i < len(fronts)
            and len(new_population) + len(fronts[i])
            <= self.population_size
        ):

            calculate_crowding_distance(fronts[i])

            new_population.extend(fronts[i])

            i += 1

        # Partial front
        if len(new_population) < self.population_size:

            remaining = (
                self.population_size
                - len(new_population)
            )

            calculate_crowding_distance(fronts[i])

            fronts[i].sort(
                key=lambda individual:
                    individual.crowding_distance,
                reverse=True
            )

            new_population.extend(
                fronts[i][:remaining]
            )

        return new_population

    def run(self):

        # Step 1: initialize
        population = self.initialize_population()

        # Step 2: evaluate
        self.evaluate_population(population)

        # Initial rank + crowding distance
        self.assign_rank_and_crowding(population)

        # Evolution
        for generation in range(self.generations):

            # Create offspring
            offspring = self.create_offspring(population)

            # Combine parent + offspring
            combined_population = (
                population + offspring
            )

            # Environmental selection
            population = self.environmental_selection(
                combined_population
            )

        # Final sorting
        fronts = fast_non_dominated_sort(population)

        return population, fronts[0]