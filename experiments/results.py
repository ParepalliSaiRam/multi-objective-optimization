import csv
import os

def save_results(
    results,
    filename="results/data/results.csv"
):

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True
    )

    file_exists = os.path.isfile(filename)

    with open(
        filename,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "benchmark",
                "algorithm",
                "seed",
                "hypervolume",
                "igd",
                "gd",
                "spacing",
                "runtime"
            ])

        for result in results:

            writer.writerow([
                result["benchmark"],
                result["algorithm"],
                result["seed"],
                result["hypervolume"],
                result["igd"],
                result["gd"],
                result["spacing"],
                result["runtime"]
            ])