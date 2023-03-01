from mesa import Model
from mesa.time import SimultaneousActivation
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector

from agents import Particle
from collector import get_info, get_worst_fitness, get_best_fitness

import numpy as np
import random


class PSO(Model):
    def __init__(self, problem, num_agents=20, agent_params=None):
        super().__init__()
        # Read parameters
        self.problem = problem
        self.num_agents = num_agents
        self.agent_params = agent_params

        self.datacollector = None
        self.best_fitness = None
        self.best_position = None
        self.grid = None
        self.schedule = None
        self.swarm = None

        self.reset()

    def reset(self) -> None:
        # Initialise the search space and schedule
        self.grid = ContinuousSpace(x_min=-1., x_max=1., y_min=-1., y_max=1., torus=True)
        self.schedule = SimultaneousActivation(self)

        # Initialise the particle swarm
        initial_positions = np.random.uniform(low=-1.0, high=1.0,
                                              size=(self.num_agents, 2))
        self.swarm = []
        for i in range(self.num_agents):
            agent = Particle(i, self, initial_positions[i, :])
            self.schedule.add(agent)
            self.grid.place_agent(agent, initial_positions[i, :])

            self.swarm.append(agent)

        self.datacollector = DataCollector(
            model_reporters={"Best Fitness": get_best_fitness, "Worst Fitness": get_worst_fitness},
            agent_reporters={"Info": get_info}
        )

    def step(self) -> None:
        # Update best and worst fitness values and positions
        fitness_values = [agent.fitness for agent in self.swarm]
        min_arg = np.argmin(fitness_values)
        self.best_position = self.swarm[min_arg].pos
        self.best_fitness = fitness_values[min_arg]

        self.datacollector.collect(self)
        self.schedule.step()

