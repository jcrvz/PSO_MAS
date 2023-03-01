from mesa import Agent
import numpy as np

__all__ = ['Particle']


class Particle(Agent):
    KINDS = ('regular', 'best', 'worst')
    DEFAULT_PARAMETERS = {
        "inertia_coefficient": 0.7,
        "self_confidence": 2.54,
        "swarm_confidence": 2.56
    }

    def __init__(self, unique_id: int, model, initial_position: tuple, **params):
        super().__init__(unique_id, model)

        self.pos = self.model.rescale_to_grid(initial_position)
        self.space_position = initial_position
        self.best_position = np.array(self.pos)
        self.velocity = 0
        self.fitness = self._get_fitness(self.space_position)
        self.best_fitness = self.fitness
        self.next_pos = None
        self.next_space_position = None
        self.kind = 'regular'

        self.inertia_coefficient = params.get("inertia_coefficient",
                                              self.DEFAULT_PARAMETERS["inertia_coefficient"])
        self.self_confidence = params.get("self_confidence",
                                          self.DEFAULT_PARAMETERS["self_confidence"])
        self.swarm_confidence = params.get("swarm_confidence",
                                           self.DEFAULT_PARAMETERS["swarm_confidence"])

    def step(self) -> None:
        current_position = self.space_position

        delta_particular_position = self.best_position - current_position
        delta_global_position = self.model.best_position - current_position

        r_1 = np.random.rand(2)
        r_2 = np.random.rand(2)

        self.velocity = self.inertia_coefficient * self.velocity + \
                        r_1 * self.self_confidence * delta_particular_position + \
                        r_2 * self.swarm_confidence * delta_global_position

        next_space_position = current_position + self.velocity
        next_pos = self.model.rescale_to_grid(next_space_position)

        if self.model.grid.out_of_bounds(next_pos):
            self.next_pos = self.model.grid.torus_adj(next_pos)
            self.next_space_position = self.model.rescale_from_grid(self.next_pos)
        else:
            self.next_pos = next_pos
            self.next_space_position = next_space_position

    def advance(self) -> None:
        self.model.grid.move_agent(self, self.next_pos)
        self.space_position = self.next_space_position
        self.fitness = self._get_fitness(self.space_position)

        # Update particular best fitness
        if self.fitness < self.best_fitness:
            self.best_position = self.space_position
            self.best_fitness = self.fitness

    def _get_fitness(self, space_position=None) -> float:
        # grid_position = position if position is not None else self.pos
        # from [0, W] to [-1, 1]
        # space_position = np.array(grid_position) / self.model.space_quant
        # from [-1, 1] to [x_min, x_max]
        real_position = self.model.problem.rescale_from_space(space_position)
        return self.model.problem.get_function_value(real_position)


class Field(Agent):
    def __init__(self, unique_id: int, model):
        super().__init__(unique_id, model)
        self.fitness_value = None
        # self.pos = self.model.rescale_to_grid(position)
        # self.space_position = position
        # real_position = self.model.problem.rescale_from_space(position)
        # return self.model.problem.get_function_value(real_position)
