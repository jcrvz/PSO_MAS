from mesa.model import Model
from mesa.time import SimultaneousActivation
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector

from .collector import get_info, get_avg_fitness, get_best_fitness

from .agents import Particle, Field
import pso_mas.landscapes as landscapes
import numpy as np


class PSO(Model):
    def __init__(self, problem, num_agents=20, muestras_por_dim=50, **kwargs):
        super().__init__()

        # Read parameters
        self.problem_name = problem
        self.num_agents = num_agents
        # self.agent_params = {} if agent_params is None else agent_params

        self.datacollector = None
        self.best_fitness = np.inf
        self.best_position = None
        self.worst_fitness = -np.inf
        self.worst_position = None
        self.space = None
        self.schedule = None
        self.swarm = None

        # Initialise the search space and schedule
        # self.space = ContinuousSpace(x_min=-1., x_max=1., y_min=-1., y_max=1., torus=True)
        self.space = ContinuousSpace(x_min=-1, x_max=1, y_min=-1, y_max=1, torus=True)
        self.schedule = SimultaneousActivation(self)

        if self.problem_name == "Sphere":
            self.problem = landscapes.Sphere(2)
        elif self.problem_name == "Rastrigin":
            self.problem = landscapes.Rastrigin(2)
        elif self.problem_name == "Weierstrass":
            self.problem = landscapes.Weierstrass(2)

        else:
            raise AttributeError("This problem has not been implemented")
        self.landscape, zmatrix = self.problem.get_landscape(muestras_por_dim)
        self.landscape_zmin = np.min(zmatrix)
        self.landscape_zmax = np.max(zmatrix)

        # Draw landscape
        for i, (pos, zval) in enumerate(self.landscape.items()):
            field = Field(int(f"{self.num_agents}0{i}"), self, zval)
            self.schedule.add(field)
            self.space.place_agent(field, pos)

        # Initialise the particle swarm
        initial_positions = np.random.uniform(low=-1.0, high=1.0, size=(self.num_agents, 2))
        self.swarm = []
        for i in range(self.num_agents):
            particle = Particle(i, self, initial_positions[i, :], **kwargs)
            self.schedule.add(particle)
            self.space.place_agent(particle, initial_positions[i, :])
            self.swarm.append(particle)

        self.get_bests()

        self.datacollector = DataCollector(
            model_reporters={"Best Fitness": get_best_fitness, "Avg. Fitness": get_avg_fitness},
            agent_reporters={"Info": get_info}
        )

    def get_bests(self):
        # Find particular bests
        #for particle in self.swarm:
        #    particle.evaluate_position()

        # Find the global best
        current_best_fitness_values = [agent.best_fitness for agent in self.swarm]
        _arg = np.argmin(current_best_fitness_values)
        best_current_fitness = current_best_fitness_values[_arg]
        best_current_position = self.swarm[_arg].best_position

        if best_current_fitness <= self.best_fitness:
            self.swarm[_arg].kind = 'best'
            self.best_fitness = current_best_fitness_values[_arg]
            self.best_position = best_current_position

    def step(self) -> None:
        self.schedule.step()
        self.get_bests()
        self.datacollector.collect(self)

