import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from desdeo_problem import MOProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.EAs.AutoNSGAII import AutoNSGAII
from desdeo_emo.utilities.samplingIGD import samplingIGD
from desdeo_emo.utilities.preprocessing_asf import preprocessing_asf
from desdeo_emo.utilities.cal_metric import cal_metric

OBJ=2
ID=2
problem_name = "ZDT2"
problem = test_problem_builder(problem_name)
no_layers = 2                  # number of layers
no_gaps   = [3, 2]             # specify the # of divisions on each layer
shrink_factors = [1.0, 0.5]    # shrinkage factor for each layer
# only useful for the many-objective scenario (i.e., objDim > 3)
igdsamSize = 10000

evolver = AutoNSGAII(
    problem,
    seed=1880184602,
    n_iterations=1,
    n_gen_per_iter=300,
    population_size=300,
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
#print(f"Number of non-dominated solutions: {len(evolver.non_dominated['objectives'])}")

#pref, plot = evolver.iterate(pref)


# obj = evolver.non_dominated["objectives"]
i2, obj_values = evolver.end()

if len(obj_values)>0:
    PF = samplingIGD(OBJ, no_layers, no_gaps, shrink_factors, igdsamSize, ID)
    PFsize   = np.shape(PF)[0]
    AutoNSGA2_Size = np.shape(obj_values)[0]
    AutoNSGA2_IGD   = cal_metric(obj_values, PF, None, AutoNSGA2_Size, PFsize)
    print(AutoNSGA2_IGD)
else:
    AutoNSGA2_IGD = 0


plt.rcParams["figure.figsize"] = [12, 8]

# should select small set of solutions to show to DM. For now we show all.
# plt.scatter(x=objective_values[:, 0], y=objective_values[:, 1], label="IBEA Front")
plt.scatter(x=obj_values[:, 0], y=obj_values[:, 1], label="NSGA-II Front iter 1")
#plt.scatter(x=responses[0][0], y=responses[0][1], label="Ref point 1")
# plt.scatter(x=obj[index][0], y=obj[index][1], label="Best solution iteration 1")
plt.title(f"Fronts")
plt.xlabel("F1")
plt.ylabel("F2")
plt.legend()
plt.show()