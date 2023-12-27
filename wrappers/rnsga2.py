import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from desdeo_problem import MOProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.EAs.RNSGAII import RNSGAII




problem_name = "ZDT2"
problem = test_problem_builder(problem_name)


evolver = RNSGAII(
    problem,
    n_iterations=1,
    n_gen_per_iter=100,
    population_size=100,
    interact=True,
    epsilon=0.001,
    normalization="front",
    weights=None,
    extreme_points_as_reference_points=False,
)

# print(evolver.allowable_interaction_types)

evolver.set_interaction_type("Reference point")
responses = np.asarray([[0.5, 0.5]])

pref, plot = evolver.start()
# response = evolver.population.ideal_fitness_val + [0.04, 0.04, 0.4]
pref.response = pd.DataFrame(
    [responses[0]], columns=pref.content["dimensions_data"].columns
)

pref, plot = evolver.iterate(pref)


# obj = evolver.non_dominated["objectives"]
i2, obj = evolver.end()

plt.rcParams["figure.figsize"] = [12, 8]

# should select small set of solutions to show to DM. For now we show all.
# plt.scatter(x=objective_values[:, 0], y=objective_values[:, 1], label="IBEA Front")
plt.scatter(x=obj[:, 0], y=obj[:, 1], label="NSGA-II Front iter 1")
plt.scatter(x=responses[0][0], y=responses[0][1], label="Ref point 1")
# plt.scatter(x=obj[index][0], y=obj[index][1], label="Best solution iteration 1")
plt.title(f"Fronts")
plt.xlabel("F1")
plt.ylabel("F2")
plt.legend()
plt.show()
