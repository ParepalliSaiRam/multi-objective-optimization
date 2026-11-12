class Individual:
    def __init__(self, variables):
        self.variables = variables

        # Objective function values
        self.objectives = []
        self.constraint_violation = 0.0

        # NSGA-II information
        self.rank = None
        self.crowding_distance = 0.0

        # Used during fast non-dominated sorting
        self.domination_count = 0
        self.dominated_solutions = []

    def __repr__(self):
        return (
            f"Individual("
            f"variables={self.variables}, "
            f"objectives={self.objectives}, "
            f"rank={self.rank}, "
            f"crowding={self.crowding_distance}"
            f")"
        )
