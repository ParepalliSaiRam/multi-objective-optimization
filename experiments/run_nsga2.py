import random
import matplotlib.pyplot as plt

from algorithms.nsga2 import NSGA2

from benchmarks.zdt import (
    zdt1,
    zdt1_bounds,
    zdt1_pareto_front,

    zdt2,
    zdt2_bounds,
    zdt2_pareto_front,

    zdt3,
    zdt3_bounds,
    zdt3_pareto_front,

    zdt4,
    zdt4_bounds,
    zdt4_pareto_front,

    zdt6,
    zdt6_bounds,
    zdt6_pareto_front
)


def run_experiment(
    objective_function,
    bounds_function,
    pareto_front_function,
    benchmark_name
):

    # Fixed seed for reproducibility
    random.seed(42)

    number_of_variables = 30

    lower_bounds, upper_bounds = bounds_function(
        number_of_variables
    )

    algorithm = NSGA2(
        population_size=100,
        generations=250,
        number_of_variables=number_of_variables,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        objective_function=objective_function,
        crossover_probability=0.9,
        mutation_probability=1.0 / number_of_variables,
        eta_c=20,
        eta_m=20
    )

    population, pareto_front = algorithm.run()

    # NSGA-II obtained front
    f1 = [
        individual.objectives[0]
        for individual in pareto_front
    ]

    f2 = [
        individual.objectives[1]
        for individual in pareto_front
    ]

    # True Pareto front
    true_front = pareto_front_function()

    # Plot
    plt.figure(figsize=(8, 6))

    for i, (true_f1, true_f2) in enumerate(true_front):

        if i == 0:
            plt.plot(
                true_f1,
                true_f2,
                label="True Pareto Front"
            )
        else:
            plt.plot(
                true_f1,
                true_f2
            )

    plt.scatter(
        f1,
        f2,
        s=20,
        label="NSGA-II"
    )

    plt.xlabel("f1")
    plt.ylabel("f2")

    plt.title(
        f"NSGA-II on {benchmark_name}"
    )

    plt.legend()
    plt.grid(True)

    plt.show()


def main():

    run_experiment(
        zdt3,
        zdt3_bounds,
        zdt3_pareto_front,
        "ZDT3"
    )


if __name__ == "__main__":
    main()