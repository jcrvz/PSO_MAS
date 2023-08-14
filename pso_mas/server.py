import mesa
from .agents import Particle, Field
from .model import PSO
from .SimpleContinuousModule import SimpleCanvas
import matplotlib as mlp


def get_rgb_color(z_val, z_min, z_max):
    z_adjusted = (z_val - z_min) / (z_max - z_min)
    return mlp.colors.to_hex(mlp.cm.GnBu_r(z_adjusted))


def pso_portrayal(agent):
    portrayal = {"Filled": "true"}

    # if agent is None:
    #    portrayal["Color"] = "#A8A8A8"
    #    portrayal["Shape"] = "rect"
    #    portrayal["h"] = 1
    #    portrayal["w"] = 1
    #    portrayal["Layer"] = 0

    if isinstance(agent, Particle):
        portrayal = {"Shape": "circle", "r": 3, "Filled": "true", "Layer": 1}
        if agent.kind == "particular_best":
            portrayal["Color"] = "red"
        elif agent.kind == "best":
            portrayal["Color"] = "yellow"
        else:  # regular
            portrayal["Color"] = "black"

    elif isinstance(agent, Field):
        color_in_hex = get_rgb_color(agent.fitness_value,
                                     agent.model.landscape_zmin, agent.model.landscape_zmax)
        portrayal = {"Shape": "rect", "h": 0.01, "w": 0.01, "Filled": "true", "Layer": 0,
                     "Color": f"{color_in_hex}"}

    return portrayal


NUM_AGENTS = 30
MAX_ITERATIONS = 10000
LENGTH = 125

model_params = {
    "num_agents": mesa.visualization.Slider(
        "Number of particles",
        NUM_AGENTS,
        2,
        200,
        2,
        description="Escoge cuántas partículas deseas implementar en el modelo",
    ),
    "problem": mesa.visualization.Choice(
        "Problem to Solve",
        "Sphere",
        ["Sphere", "Rastrigin", "Weierstrass"],
        "Selecciona el problema a resolver"
    ),
    "pso_variant": mesa.visualization.Choice(
        "Variante de PSO", "Inertial",
        ["Inertial", "Constriction"],
        "Se activa la variante de PSO"
    ),
    "velocity_coefficient": mesa.visualization.NumberInput(
        "Coef. de Inercia/Constricción", 0.6,
        "Indica el valor del coeficiente relacionado con la variante de PSO"
    ),
    "self_confidence": mesa.visualization.NumberInput(
        "Coef. de Confianza Propia", 1.56,
        "Indica el valor del coeficiente de confianza propia"
    ),
    "swarm_confidence": mesa.visualization.NumberInput(
        "Coef. de Confianza en el Enjambre", 1.56,
        "Indica el valor del coeficiente de confianza en el enjambre"),
    "greedy_selection": mesa.visualization.Choice(
        "Mecanismo de Selección", "Direct",
        ["Direct", "Greedy"],
        "Se activa la selección greedy, es decir, solo acepta soluciones que mejoran"
    ),
    "operator": mesa.visualization.Choice(
        "Operator", "Particle Swarm",
        ["Particle Swarm", "Spiral"],
        "Se selecciona el tipo de perturbaciones"
    ),
    "muestras_por_dim": mesa.visualization.Slider(
        "Muestras por Dimensións",
        100,
        10,
        200,
        10,
        description="Escoge cuántas muestras por dimensión usar para ver el panorama",
    ),
}

canvas_element = SimpleCanvas(pso_portrayal, 500, 500)
best_evolution = mesa.visualization.ChartModule(
    [{"Label": "Best Fitness", "Color": "blue", "label": "Best"}])
avg_evolution = mesa.visualization.ChartModule(
    [{"Label": "Avg. Fitness", "Color": "red", "label": "Average"}])

server = mesa.visualization.ModularServer(
    model_cls=PSO,
    visualization_elements=[canvas_element, best_evolution, avg_evolution],
    name="PSO MAS", model_params=model_params, port=8521
)
server.max_steps = MAX_ITERATIONS
