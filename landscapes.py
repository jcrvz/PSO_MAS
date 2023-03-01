import numpy as np

__all__ = ['Sphere', 'Rastrigin']


class _BasicProblem:
    """
    This is the basic class for a generic optimisation problem.
    """

    def __init__(self, variable_num):
        """
        Initialise a problem object using only the dimensionality of its domain.

        :param int variable_num: optional.
            Number of dimensions or variables for the problem domain. The default values is 2 (this is the common option
            for plotting purposes).
        """
        self.variable_num = variable_num
        self.max_search_range = None
        self.min_search_range = None
        self.span_search_range = None
        self.centre_search_range = None
        self.set_boundaries([0] * self.variable_num, [0] * self.variable_num)

        self.optimal_solution = np.array([0] * self.variable_num)
        self.optimal_fitness = 0
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

    def set_boundaries(self, min_search_range=None, max_search_range=None) -> None:
        if min_search_range is not None:
            self.min_search_range = np.array(min_search_range) if isinstance(
                min_search_range, list) else min_search_range

        if max_search_range is not None:
            self.max_search_range = np.array(max_search_range) if isinstance(
                max_search_range, list) else max_search_range

        if not (min_search_range is None and max_search_range is None):
            self.span_search_range = self.max_search_range - self.min_search_range
            self.centre_search_range = 0.5 * (self.max_search_range + self.min_search_range)

    def get_landscape(self, samples_per_dimension=50) -> np.ndarray:
        x = np.linspace(self.min_search_range[0], self.max_search_range[0],
                        samples_per_dimension)
        y = np.linspace(self.min_search_range[1], self.max_search_range[1],
                        samples_per_dimension)

        # Create the grid matrices
        matrix_x, matrix_y = np.meshgrid(x, y)

        # Evaluate each node of the grid into the problem function
        matrix_z = []
        for xy_list in zip(matrix_x, matrix_y):
            z = []
            for xy_input in zip(xy_list[0], xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[2:self.variable_num]))
                z.append(self.get_function_value(np.array(tmp)))
            matrix_z.append(z)
        matrix_z = np.array(matrix_z)

        return matrix_z

    def rescale_from_space(self, position: np.ndarray | list | tuple) -> np.ndarray:
        # From [-1, 1] to [x_min, x_max]
        return 0.5 * np.array(position) * self.span_search_range + self.centre_search_range

    def rescale_to_space(self, position: np.ndarray | list | tuple) -> np.ndarray:
        # From [x_min, x_max] to [-1, 1]
        return 2 * (np.array(position) - self.centre_search_range) / self.span_search_range

    def get_function_value(self, variables) -> float:
        return np.nan


class Sphere(_BasicProblem):
    def __init__(self, variable_num=2):
        super().__init__(variable_num)
        self.variable_num = variable_num
        self.set_boundaries([-100.] * self.variable_num, [100.] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Sphere'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': True,
                         'Convex': True}

        self.landscape = self.get_landscape()

    def get_function_value(self, variables):
        return np.sum(np.square(variables))


class Rastrigin(_BasicProblem):
    def __init__(self, variable_num=2):
        super().__init__(variable_num)
        self.variable_num = variable_num
        self.set_boundaries([-5.12] * self.variable_num, [5.12] * self.variable_num)
        self.max_search_range = np.array([5.12] * self.variable_num)
        self.min_search_range = np.array([-5.12] * self.variable_num)
        self.optimal_solution = np.array([0.] * self.variable_num)
        self.global_optimum_solution = 0.
        self.func_name = 'Rastrigin'
        self.features = {'Continuous': True,
                         'Differentiable': True,
                         'Separable': True,
                         'Scalable': True,
                         'Unimodal': False,
                         'Convex': False}

        self.landscape = self.get_landscape()

    def get_function_value(self, variables):
        return 10. * self.variable_num + np.sum(
            np.square(variables) - 10. * np.cos(2. * np.pi * variables))
