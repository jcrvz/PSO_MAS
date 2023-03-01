from model import PSO
from landscapes import Sphere


problem = Sphere()

pso = PSO(problem, 20)

pso.run_model()
