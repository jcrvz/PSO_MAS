import numpy as np
from mesa.datacollection import DataCollector

__all__ = ["get_best_fitness", "get_worst_fitness", "get_info"]

from typing import Tuple


def get_best_fitness(model) -> float:
    return model.best_fitness


def get_avg_fitness(model) -> float:
    return np.average([particle.fitness for particle in model.swarm])


def get_worst_fitness(model) -> float:
    return model.worst_fitness


def get_info(agent) -> Tuple[str, tuple]:
    return agent.kind, agent.pos
