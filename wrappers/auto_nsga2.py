import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from desdeo_problem import MOProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.EAs.AutoNSGAII import AutoNSGAII

problem_name = "ZDT2"
problem = test_problem_builder(problem_name)


evolver = AutoNSGAII(
    problem,
    seed=1880184602,
    n_iterations=1,
    n_gen_per_iter=300,
    population_size=100,
    selection_parents = "tournament",
    slection_tournament_size = 2,
    crossover = "BLX_ALPHA",
    crossover_probability = 0.0318,
    blx_alpha_crossover = 0.7874,
    crossover_repair = "bounds",
    mutation = "polynomial",
    mutation_probability = 0.0328,
    mutation_repair = "round",
    polinomial_mut_dist_index  = 140.3278,
)

# print(evolver.allowable_interaction_types)

#evolver.set_interaction_type("Reference point")
#responses = np.asarray([[0.5, 0.5]])

#pref, plot = evolver.start()
# response = evolver.population.ideal_fitness_val + [0.04, 0.04, 0.4]
#pref.response = pd.DataFrame(
#    [responses[0]], columns=pref.content["dimensions_data"].columns
#)
evolver.start()
while evolver.continue_evolution():
    evolver.iterate()
    print(f"Running iteration {evolver._iteration_counter}")
#print(f"Number of non-dominated solutions: {len(evolver.non_dominated['objectives'])}")

#pref, plot = evolver.iterate(pref)


# obj = evolver.non_dominated["objectives"]
i2, obj = evolver.end()

plt.rcParams["figure.figsize"] = [12, 8]

# should select small set of solutions to show to DM. For now we show all.
# plt.scatter(x=objective_values[:, 0], y=objective_values[:, 1], label="IBEA Front")
plt.scatter(x=obj[:, 0], y=obj[:, 1], label="NSGA-II Front iter 1")
#plt.scatter(x=responses[0][0], y=responses[0][1], label="Ref point 1")
# plt.scatter(x=obj[index][0], y=obj[index][1], label="Best solution iteration 1")
plt.title(f"Fronts")
plt.xlabel("F1")
plt.ylabel("F2")
plt.legend()
plt.show()
