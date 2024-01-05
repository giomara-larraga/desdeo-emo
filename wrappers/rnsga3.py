import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from desdeo_problem import MOProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.EAs.RNSGAIII import RNSGAIII

problem_name = "ZDT1"
problem = test_problem_builder(problem_name)


evolver = RNSGAIII(
    problem,
    n_iterations=1,
    n_gen_per_iter=100,
    population_size=100,
    interact=True,
    mu=0.5,
    save_non_dominated=True,
    seed = 45,
    selection_parents = "random",
    slection_tournament_size = 2,
    crossover = "SBX",
    crossover_probability = 1,
    crossover_distribution_index = 0.5,
    crossover_repair = "bounds",
    mutation = "uniform",
    mutation_probability = 1/20,
    mutation_repair = "bounds",
    uniform_mut_perturbation  = 20,
)

evolver.set_interaction_type("Reference point")
responses = np.asarray([[0.5, 0.5]])

pref, plot = evolver.start()
# response = evolver.population.ideal_fitness_val + [0.04, 0.04, 0.4]
pref.response = pd.DataFrame(
    [responses[0]], columns=pref.content["dimensions_data"].columns
)

pref, plot = evolver.iterate(pref)


# obj = evolver.non_dominated["objectives"]
i2, obj, base = evolver.end()

plt.rcParams["figure.figsize"] = [12, 8]

# should select small set of solutions to show to DM. For now we show all.
# plt.scatter(x=objective_values[:, 0], y=objective_values[:, 1], label="IBEA Front")
plt.scatter(x=obj[:, 0], y=obj[:, 1], label="RNSGA-III Front iter 1")
plt.scatter(x=responses[0][0], y=responses[0][1], label="Ref point 1")
# plt.scatter(x=obj[index][0], y=obj[index][1], label="Best solution iteration 1")
plt.title(f"Fronts")
plt.xlabel("F1")
plt.ylabel("F2")
plt.legend()
plt.show()
