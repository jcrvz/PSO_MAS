from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from agents import Car, TrafficLight, Field
from model import CrossRoad


def cross_road_portrayal(agent):
    portrayal = {"Filled": "true"}

    # Street
    if agent is None:
        portrayal["Color"] = "#A8A8A8"
        portrayal["Shape"] = "rect"
        portrayal["h"] = 1
        portrayal["w"] = 1
        portrayal["Layer"] = 0

    if type(agent) is Car:
        if agent.colour == "orange":
            portrayal["Color"] = "#FF9C38"
        elif agent.colour == "blue":
            portrayal["Color"] = "#0000FF"
        elif agent.colour == "purple":
            portrayal["Color"] = "#D847FF"
        else:  # black
            portrayal["Color"] = "#000000"
        portrayal["Shape"] = "circle"
        portrayal["r"] = 1.1
        portrayal["Layer"] = 1
        if agent.waiting == 1:
            portrayal["text"] = "x"
            portrayal["text_color"] = "Red"
        else:
            portrayal["text"] = ""
            portrayal["text_color"] = "Green"

    elif type(agent) is TrafficLight:
        if agent.colour == "green":
            portrayal["Color"] = "#00FF00"
        else:  # red
            portrayal["Color"] = "#FF0000"
        portrayal["Shape"] = "circle"
        portrayal["r"] = 1
        portrayal["Layer"] = 1

    elif type(agent) is Field:
        if agent.colour == 'brown':
            portrayal["Color"] = "#865700"
        elif agent.colour == 'olive':
            portrayal["Color"] = "#828232"
        else:  # dark green
            portrayal["Color"] = "#0A6414"
        portrayal["Shape"] = "rect"
        portrayal["h"] = 1
        portrayal["w"] = 1
        portrayal["Layer"] = 0

    return portrayal

NUM_CARS = 50
HALF_LENGTH = 20
TRAFFIC_TIMER = 10
CAR_TURNING_RATE = 0.5
MAX_ITERATIONS = 500

pixel_ratio = 10

model_params = dict(
    num_agents=UserSettableParameter(
        "slider", "Number of cars", NUM_CARS, 5, 200, 5),
    half_length=UserSettableParameter(
        "slider", "Half Length", HALF_LENGTH, 5, 50, 5),
    traffic_time=UserSettableParameter(
        "slider", "Traffic Timer", TRAFFIC_TIMER, 5, 100, 5),
    car_turning_rate=UserSettableParameter(
        "slider", "Turning Rate", CAR_TURNING_RATE, 0.0, 1.0, 0.1),
    )

length = model_params['half_length'].value * 2
canvas_element = CanvasGrid(
    cross_road_portrayal, length, length,
    pixel_ratio * length, pixel_ratio * length)
chart_element = ChartModule([
    {"Label": "Waiting", "Color": "#AA0000"},
    {"Label": "Running", "Color": "#00AA00"}
])


server = ModularServer(model_cls=CrossRoad,
                       visualization_elements=[canvas_element, chart_element],
                       name="Cross Road", model_params=model_params
                       )
server.max_steps = MAX_ITERATIONS
server.launch()
