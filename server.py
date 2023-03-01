from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import Slider

from agents import Particle, Field
from model import PSO
from landscapes import Sphere, Rastrigin


def pso_portrayal(agent):
    portrayal = {"Filled": "true"}

    if agent is None:
        portrayal["Color"] = "#A8A8A8"
        portrayal["Shape"] = "rect"
        portrayal["h"] = 1
        portrayal["w"] = 1
        portrayal["Layer"] = 0

    if isinstance(agent, Particle):
        if agent.kind == "worst":
            portrayal["Color"] = "#0000FF"
        elif agent.kind == "best":
            portrayal["Color"] = "#D847FF"
        else:  # regular
            portrayal["Color"] = "#000000"
        portrayal["Shape"] = "circle"
        portrayal["r"] = 1.1
        portrayal["Layer"] = 1

    return portrayal


NUM_AGENTS = 20
MAX_ITERATIONS = 10000
LENGTH = 100

model_params = dict(
    problem=Sphere(),
    num_agents=Slider("Number of particles", NUM_AGENTS, 2, 200, 20),
    space_quant=LENGTH,
    # space_quant=Slider("Samples per dimension", LENGTH, 10, 500, 10),
)
# iterations = Slider("Number of iterations", MAX_ITERATIONS, 10, 1000, 10)

canvas_element = CanvasGrid(pso_portrayal,
                            LENGTH, LENGTH, 500, 500)
best_evolution = ChartModule([{"Label": "Best Fitness", "Color": "#AA0000"}],
                             canvas_height=75, canvas_width=300)
worst_evolution = ChartModule([{"Label": "Worst Fitness", "Color": "#00AA00"}],
                              canvas_height=75, canvas_width=300)



server = ModularServer(model_cls=PSO,
                       visualization_elements=[canvas_element, best_evolution, worst_evolution],
                       name="PSO MAS", model_params=model_params
                       )
server.max_steps = MAX_ITERATIONS
server.launch()
