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

        self.pos = initial_position
        self.best_position = np.array(self.pos)
        self.velocity = 0
        self.fitness = self._get_fitness(self.best_position)
        self.best_fitness = np.inf
        self.kind = 'regular'

        self.inertia_coefficient = params.get(
            "inertia_coefficient", self.DEFAULT_PARAMETERS["inertia_coefficient"])
        self.self_confidence = params.get(
            "self_confidence", self.DEFAULT_PARAMETERS["self_confidence"])
        self.swarm_confidence = params.get(
            "swarm_confidence", self.DEFAULT_PARAMETERS["swarm_confidence"])

    def step(self) -> None:
        current_position = np.array(self.pos)
        self.fitness = self._get_fitness(current_position)

        if self.fitness <= self.best_fitness:
            self.best_position = current_position
            self.best_fitness = self.fitness

        delta_particular_position = self.best_position - current_position
        delta_global_position = self.model.best_position - current_position

        r_1 = np.random.rand(2)
        r_2 = np.random.rand(2)

        self.velocity = self.inertia_coefficient * self.velocity + \
                        r_1 * self.self_confidence * delta_particular_position + \
                        r_2 * self.swarm_confidence * delta_global_position

        next_pos = current_position + self.velocity

        if self.model.space.out_of_bounds(next_pos):
            for i, pos_element in enumerate(next_pos):
                if pos_element >= 1.:
                    pos_element = np.random.uniform(0.9, 1.0)
                elif pos_element <= -1.:
                    pos_element = np.random.uniform(-1.0, -0.1)
                next_pos[i] = pos_element

    #def advance(self) -> None:
    #    self.next_pos = next_pos

        self.model.space.move_agent(self, next_pos)


    def _get_fitness(self, position=None) -> float:
        return self.model.problem.get_function_value(position)


class Field(Agent):
    def __init__(self, unique_id: int, model, fitness_value=None):
        super().__init__(unique_id, model)
        self.fitness_value = fitness_value
        self.kind = 'landscape'
