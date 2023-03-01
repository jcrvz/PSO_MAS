from mesa.datacollection import DataCollector

__all__ = ["get_best_fitness", "get_worst_fitness", "get_info"]

def get_best_fitness(model) -> float:
    return model.best_fitness


def get_worst_fitness(model) -> float:
    return model.worst_fitness


def get_info(agent) -> tuple[str, tuple]:
    return agent.kind, agent.pos
