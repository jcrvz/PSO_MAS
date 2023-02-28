from mesa import Model
from mesa.time import SimultaneousActivation
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector

from agents import Particle

import numpy as np
import random

class PSO(Model):
    def __init__(self, problem, num_agents=20, agent_params=None):
        super().__init__()
        # Read parameters
        self.problem = problem
        self.num_agents = num_agents
        self.agent_params = agent_params

        self.best_position = None
        self.grid = None
        self.schedule = None
        self.reset()

    def reset(self) -> None:
        # Initialise the search space and schedule
        self.grid = ContinuousSpace(x_min=-1., x_max=1., y_min=-1., y_max=1., torus=True)
        self.schedule = SimultaneousActivation(self)

        # Initialise the particle swarm
        initial_positions = np.random.uniform(low=-1.0, high=1.0,
                                              size=(self.num_agents, 2))
        for i in range(self.num_agents):
            agent = Particle(i, self, **self.agent_params)
            self.schedule.add(agent)
            self.grid.place_agent(agent, initial_positions[i, :])


