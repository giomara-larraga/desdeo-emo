import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from desdeo_problem import MOProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.EAs.PBEA import PBEA
from desdeo_tools.utilities.quality_indicator import preference_indicator

problem_name = "WFG1"
problem = test_problem_builder(problem_name, n_of_objectives=2)


delta = 0.02
evolver = PBEA(
    problem,
    interact=True,
    population_size=20,
    initial_population=None,
    n_iterations=1,
    n_gen_per_iter=500,
    total_function_evaluations=6000,
    indicator=preference_indicator,
    delta=delta,
)

responses = np.asarray([[0.1, 0.1]])
pref, plot = evolver.requests()

pref.response = pd.DataFrame(
    [responses[0]], columns=pref.content["dimensions_data"].columns
)
pref, plot = evolver.iterate(pref)
i2, obj = evolver.end()

plt.rcParams["figure.figsize"] = [12, 8]

# should select small set of solutions to show to DM. For now we show all.
# plt.scatter(x=objective_values[:, 0], y=objective_values[:, 1], label="IBEA Front")
plt.scatter(x=obj[:, 0], y=obj[:, 1], label="PBEA Front iter 1")
plt.scatter(x=responses[0][0], y=responses[0][1], label="Ref point 1")
# plt.scatter(x=obj[index][0], y=obj[index][1], label="Best solution iteration 1")
plt.title(f"Fronts")
plt.xlabel("F1")
plt.ylabel("F2")
plt.legend()
plt.show()
