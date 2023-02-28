from mesa import Agent
import numpy as np

__all__ = ['Particle']


class Particle(Agent):
    KINDS = ('regular', 'best', 'worst')

    def __init__(self, unique_id, model, **params):
        super().__init__(unique_id, model)

        self.particular_best_position = np.array(self.pos)
        self.velocity = 0
        self.next_position = None
        self.colour = 'black'

        self.inertia_coefficient = params.get('inertia_coefficient', 0.7)
        self.self_confidence = params.get('self_confidence', 2.54)
        self.swarm_confidence = params.get('swarm_confidence', 2.56)

    def step(self) -> None:
        current_position = np.array(self.pos)

        delta_particular_position = self.particular_best_position - current_position
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

    def _get_fitness(self) -> float:
        real_position = self.model.problem.rescale_to_space(self.pos)
        return self.model.problem.get_function_value(real_position)
