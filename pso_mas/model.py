from mesa.model import Model
from mesa.time import SimultaneousActivation
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector

from .collector import get_info, get_worst_fitness, get_best_fitness

from .agents import Particle, Field
import pso_mas.landscapes as landscapes
import numpy as np


class PSO(Model):
    def __init__(self, problem, num_agents=20, muestras_por_dim=50, agent_params=None, **kwargs):
        super().__init__()

        # Read parameters
        self.problem_name = problem
        self.num_agents = num_agents
        # self.agent_params = {} if agent_params is None else agent_params

        self.datacollector = None
        self.best_fitness = np.inf
        self.worst_fitness = -np.inf
        self.best_position = None
        self.worst_position = None
        self.space = None
        self.schedule = None
        self.swarm = None

        # Initialise the search space and schedule
        # self.space = ContinuousSpace(x_min=-1., x_max=1., y_min=-1., y_max=1., torus=True)
        self.space = ContinuousSpace(x_min=-1, x_max=1, y_min=-1, y_max=1, torus=False)
        self.schedule = SimultaneousActivation(self)

        if self.problem_name == "Sphere":
            self.problem = landscapes.Sphere(2)
        elif self.problem_name == "Rastrigin":
            self.problem = landscapes.Rastrigin(2)
        elif self.problem_name == "Stochastic":
            self.problem = landscapes.Stochastic(2)

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
            particle = Particle(i, self, initial_positions[i, :])
            self.schedule.add(particle)
            self.space.place_agent(particle, initial_positions[i, :])

            self.swarm.append(particle)

        self.datacollector = DataCollector(
            model_reporters={"Best Fitness": get_best_fitness,
                             "Worst Fitness": get_worst_fitness},
            agent_reporters={"Info": get_info}
        )

    def get_best(self):
        best_fitness_values = [agent.best_fitness for agent in self.swarm]
        best_arg = np.argmin(best_fitness_values)
        self.best_fitness = best_fitness_values[best_arg]
        self.best_position = np.array(self.swarm[best_arg].pos)


    def step(self) -> None:

        # Update best and worst fitness values and positions
        #best_fitness_values = [agent.best_fitness for agent in self.swarm]
        #fitness_values = [agent.fitness for agent in self.swarm]

        #worst_arg = np.argmax(best_fitness_values)


        #self.worst_fitness = best_fitness_values[worst_arg]
        #self.worst_position = self.swarm[worst_arg].pos

        #for particle in self.swarm:
        #    particle.kind = 'regular'
        #self.swarm[best_arg].kind = 'best'
        #self.swarm[worst_arg].kind = 'worst'
        self.get_best()

        self.datacollector.collect(self)
        self.schedule.step()

        #print("Best: {}, Worst: {}".format(self.best_fitness, self.worst_fitness))

