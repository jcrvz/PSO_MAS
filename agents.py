from mesa import Agent
import numpy as np

__all__ = ['Particle']


class Particle(Agent):
    KINDS = ('regular', 'best', 'worst')

    def __init__(self, unique_id, model, initial_position):
        super().__init__(unique_id, model)

        self.pos = tuple(initial_position)
        self.best_position = np.array(self.pos)
        self.velocity = 0
        self.fitness = self._get_fitness()
        self.best_fitness = self.fitness
        self.next_position = None
        self.kind = 'regular'

        self.inertia_coefficient = 0.7  # params.get('inertia_coefficient', 0.7)
        self.self_confidence = 2.54  # params.get('self_confidence', 2.54)
        self.swarm_confidence = 2.56  # params.get('swarm_confidence', 2.56)

    def step(self) -> None:
        current_position = np.array(self.pos)

        delta_particular_position = self.best_position - current_position
        delta_global_position = self.model.best_position - current_position

        r_1 = np.random.rand(2)
        r_2 = np.random.rand(2)

        self.velocity = self.inertia_coefficient * self.velocity + \
            r_1 * self.self_confidence * delta_particular_position + \
            r_2 * self.swarm_confidence * delta_global_position

        next_position = current_position + self.velocity

        if self.model.grid.out_of_bounds(next_position):
            self.next_position = self.model.grid.torus_adj(tuple(next_position))
        else:
            self.next_position = tuple(next_position)

    def advance(self) -> None:
        self.model.grid.move_agent(self, self.next_position)
        self.fitness = self._get_fitness(np.array(self.next_position))

        # Update particular best fitness
        if self.fitness <= self.best_fitness:
            self.best_position = np.array(self.pos)
            self.best_fitness = self.fitness

    def _get_fitness(self, position=None) -> float:
        space_position = position if position is not None else self.pos
        real_position = self.model.problem.rescale_to_space(space_position)
        return self.model.problem.get_function_value(real_position)
