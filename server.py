from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from agents import Particle
from model import PSO
from landscapes import Sphere, Rastrigin


def cross_road_portrayal(agent):
    portrayal = {"Filled": "true"}

    # Street
    if agent is None:
        portrayal["Color"] = "#A8A8A8"
        portrayal["Shape"] = "rect"
        portrayal["h"] = 1
        portrayal["w"] = 1
        portrayal["Layer"] = 0

    if type(agent) is Particle:
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
MAX_ITERATIONS = 500
LENGTH = 20

PIXEL_RATIO = 10

model_params = dict(
    problem=Sphere(2),
    num_agents=UserSettableParameter(
        "slider", "Number of cars", NUM_AGENTS, 5, 200, 5),
)

canvas_element = CanvasGrid(
    cross_road_portrayal, LENGTH, LENGTH,
    PIXEL_RATIO * LENGTH, PIXEL_RATIO * LENGTH)
chart_element = ChartModule([
    {"Label": "Best Fitness", "Color": "#AA0000"},
    {"Label": "Worst Fitness", "Color": "#00AA00"}
])

server = ModularServer(model_cls=PSO,
                       visualization_elements=[chart_element],
                       name="PSO MAS", model_params=model_params
                       )
server.max_steps = MAX_ITERATIONS
server.launch()
