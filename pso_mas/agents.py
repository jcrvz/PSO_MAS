from mesa import Agent
import numpy as np
from itertools import combinations as _get_combinations

__all__ = ['Particle']


class Particle(Agent):
    KINDS = ('regular', 'best', 'worst')
    DEFAULT_PARAMETERS = {
        "velocity_coefficient": 0.7,
        "self_confidence": 2.54,
        "swarm_confidence": 2.56,
        "selection": "Direct",
        "operator": "Particle Swarm",
        "pso_variant": "Inertial",
    }

    def __init__(self, unique_id: int, model, initial_position: tuple, **params):
        super().__init__(unique_id, model)

        self.velocity_coefficient: float = params.get(
            "velocity_coefficient", self.DEFAULT_PARAMETERS["velocity_coefficient"])
        self.self_confidence: float = params.get(
            "self_confidence", self.DEFAULT_PARAMETERS["self_confidence"])
        self.swarm_confidence: float = params.get(
            "swarm_confidence", self.DEFAULT_PARAMETERS["swarm_confidence"])
        self.selection: str = params.get(
            "selection", self.DEFAULT_PARAMETERS["selection"])
        self.operator: str = params.get(
            "operator", self.DEFAULT_PARAMETERS["operator"])
        self.pso_variant: str = params.get(
            "pso_variant", self.DEFAULT_PARAMETERS["pso_variant"])

        self.pos = initial_position
        self.current_position = None
        self.fitness = None
        self.best_position = None
        self.next_pos = None
        self.best_fitness = np.inf

        self.velocity = np.zeros(2)
        self.kind = 'regular'

        self.evaluate_position()

    def _get_fitness(self, position=None) -> float:
        return self.model.problem.get_function_value(position)

    def evaluate_position(self):
        # Evaluate the current position
        self.current_position = np.array(self.pos)
        self.fitness = self._get_fitness(self.current_position)

        # Find particular best position
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.current_position
            self.kind = "particular_best"
        else:
            self.kind = "regular"
            if self.selection == "Greedy":
                self.fitness = self.best_fitness
                self.current_position = self.best_position
                self.next_pos = self.current_position
                #self.pos = tuple(self.current_position)
                self.model.space.move_agent(self, tuple(self.next_pos))

    def particle_swarm_operation(self):
        # Determine the differences
        delta_particular_position = self.best_position - self.current_position
        delta_global_position = self.model.best_position - self.current_position

        particular_contrib = np.random.rand(2) * self.self_confidence * delta_particular_position
        swarm_contrib = np.random.rand(2) * self.swarm_confidence * delta_global_position

        # Determine the velocity
        if self.pso_variant == "Constriction":
            self.velocity = self.velocity_coefficient * (self.velocity + particular_contrib + swarm_contrib)
        else:  # if self.pso_variant == "Inertial":
            self.velocity = self.velocity_coefficient * self.velocity + particular_contrib + swarm_contrib

        # Determine the new position
        self.next_pos = self.current_position + self.velocity


    def spiral_operation(self):
        radius = 0.99
        angle = 12.25
        sin_val = np.sin(np.deg2rad(angle))
        cos_val = np.cos(np.deg2rad(angle))
        rotation_matrix = np.array([[cos_val, -sin_val], [sin_val, cos_val]])

        self.next_pos = (self.model.best_position + radius * np.matmul(
            rotation_matrix, (self.current_position - self.model.best_position)))

    def step(self) -> None:
        if self.operator == "Spiral":
            self.spiral_operation()
        else:  # Particle Swarm
            self.particle_swarm_operation()
            #if self.kind == "best":
            #    print(f"Current: {self.current_position}, Next: {self.next_pos}")

    def advance(self) -> None:
        self.model.space.move_agent(self, tuple(self.next_pos))

        self.evaluate_position()
        # if self.model.space.out_of_bounds(self.next_pos):
        #    for i, pos_element in enumerate(self.next_pos):
        #        if pos_element >= 1.:
        #            pos_element = np.random.uniform(0.99, 1.0)
        #        elif pos_element <= -1.:
        #            pos_element = np.random.uniform(-1.0, -0.01)
        #        self.next_pos[i] = pos_element
        # self.next_pos = tuple(self.pos)


class Field(Agent):
    def __init__(self, unique_id: int, model, fitness_value=None):
        super().__init__(unique_id, model)
        self.fitness_value = fitness_value
        self.kind = 'landscape'
