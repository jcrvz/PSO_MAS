from mesa import Model
from mesa.time import SimultaneousActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from agents import Particle, Field
from collector import get_info, get_worst_fitness, get_best_fitness

import numpy as np


class PSO(Model):
    def __init__(self, problem, num_agents=20, space_quant=25, agent_params=None, **kwargs):
        super().__init__()
        self.landscape = None
        self.space_quant = space_quant

        # Read parameters
        self.problem = problem
        self.num_agents = num_agents
        self.agent_params = {} if agent_params is None else agent_params

        self.datacollector = None
        self.best_fitness = np.inf
        self.worst_fitness = -np.inf
        self.best_position = None
        self.worst_position = None
        self.grid = None
        self.schedule = None
        self.swarm = None

        self.reset()

    def reset(self) -> None:
        # Initialise the search space and schedule
        # self.grid = ContinuousSpace(x_min=-1., x_max=1., y_min=-1., y_max=1., torus=True)
        self.grid = MultiGrid(width=self.space_quant, height=self.space_quant, torus=True)
        self.schedule = SimultaneousActivation(self)

        self.landscape = self.problem.get_landscape(samples_per_dimension=self.space_quant)

        # Initialise the field
        # for i, (content, x, y) in enumerate(self.grid.coord_iter()):
        #     field = Field(i, self)
        #     self.grid.place_agent(field, (x, y))
        #     field.fitness_value = self.landscape[x, y]

        # Initialise the particle swarm
        initial_positions = np.random.uniform(low=-1.0, high=1.0, size=(self.num_agents, 2))
        self.swarm = []
        for i in range(self.num_agents):
            particle = Particle(i + self.space_quant**2 + 1, self, initial_positions[i, :], **self.agent_params)
            self.schedule.add(particle)
            self.grid.place_agent(particle, self.rescale_to_grid(initial_positions[i, :]))

            self.swarm.append(particle)

        self.datacollector = DataCollector(
            model_reporters={"Best Fitness": get_best_fitness, "Worst Fitness": get_worst_fitness},
            agent_reporters={"Info": get_info}
        )

    def step(self) -> None:

        # Update best and worst fitness values and positions
        best_fitness_values = [agent.best_fitness for agent in self.swarm]
        fitness_values = [agent.best_fitness for agent in self.swarm]

        min_arg = np.argmin(best_fitness_values)
        max_arg = np.argmax(fitness_values)

        self.best_fitness = best_fitness_values[min_arg]
        self.worst_fitness = fitness_values[max_arg]

        self.best_position = self.swarm[min_arg].space_position
        self.worst_position = self.swarm[max_arg].space_position

        for particle in self.swarm:
            particle.kind = 'regular'
        self.swarm[min_arg].kind = 'best'
        self.swarm[max_arg].kind = 'worst'

        print("Best: {}, Worst: {}".format(self.best_fitness, self.worst_fitness))

        self.datacollector.collect(self)
        self.schedule.step()

    def rescale_from_grid(self, position: np.ndarray | list | tuple) -> np.ndarray:
        # From [0, W] to [-1, 1]
        return (2 * np.array(position) - self.space_quant) / self.space_quant

    def rescale_to_grid(self, position: np.ndarray | list | tuple) -> tuple:
        # From [-1, 1] to [0, W]
        return tuple([int((pos_comp + 1) * self.space_quant / 2) for pos_comp in position])
