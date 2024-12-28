import numpy as np
import pandas as pd
import pygmo

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from desdeo_problem import MOProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.EAs.RNSGAIII import RNSGAIII
from desdeo_emo.EAs.MOEAD import MOEA_D
from desdeo_emo.utilities.pf_samples import pf_samples
from desdeo_emo.utilities.preprocessing_asf import preprocessing_asf
from desdeo_emo.utilities.cal_metric import cal_metric
from desdeo_emo.utilities.cal_metric import cal_hv
from desdeo_problem.testproblems.RiverPollution import river_pollution_problem
from desdeo_problem import variable_builder, ScalarObjective, VectorObjective, MOProblem

#problem_name = "ZDT1"
#problem = test_problem_builder(problem_name)
# create the problem
def f_1(x):
    return 4.07 + 2.27 * x[:, 0]

def f_2(x):
    return 2.60 + 0.03*x[:, 0] + 0.02*x[:, 1] + 0.01 / (1.39 - x[:, 0]**2) + 0.30 / (1.39 - x[:, 1]**2)

def f_3(x):
    return 8.21 - 0.71 / (1.09 - x[:, 0]**2)

def f_4(x):
    return 0.96 - 0.96 / (1.09 - x[:, 1]**2)

# def f_5(x):
    # return -0.96 + 0.96 / (1.09 - x[:, 1]**2)

def f_5(x):
    return np.max([np.abs(x[:, 0] - 0.65), np.abs(x[:, 1] - 0.65)], axis=0)

def run(reference_point, generations, population_size, mu, selection, tournament_size, crossover, crossover_probability, blx_alpha_crossover, crossover_distribution_index, crossover_repair, mutation,mutation_probability, mutation_repair, uniform_mut_perturbation, polinomial_mut_dist_index):
    f1 = ScalarObjective(name="f1", evaluator=f_1, maximize=True)
    f2 = ScalarObjective(name="f2", evaluator=f_2, maximize=True)
    f3 = ScalarObjective(name="f3", evaluator=f_3, maximize=True)
    f4 = ScalarObjective(name="f4", evaluator=f_4, maximize=True)
    f5 = ScalarObjective(name="f5", evaluator=f_5, maximize=False)

    varsl = variable_builder(["x_1", "x_2"],
        initial_values=[0.5, 0.5],
        lower_bounds=[0.3, 0.3],
        upper_bounds=[1.0, 1.0]
        )

    problem = MOProblem(variables=varsl, objectives=[f1, f2, f3, f4, f5])
    evolver = RNSGAIII(
        problem,
        n_iterations=1,
        n_gen_per_iter=generations,
        population_size=None,
        interact=True,
        mu=mu,
        save_non_dominated=True,
        seed = None,
        selection_parents = selection,
        slection_tournament_size = tournament_size,
        crossover = crossover,
        crossover_probability = crossover_probability,
        crossover_distribution_index = crossover_distribution_index,
        blx_alpha_crossover = blx_alpha_crossover,
        crossover_repair = crossover_repair,
        mutation = mutation,
        mutation_probability = mutation_probability,
        mutation_repair = mutation_repair,
        uniform_mut_perturbation  = uniform_mut_perturbation,
        polinomial_mut_dist_index = polinomial_mut_dist_index,
    )

    evolver.set_interaction_type("Reference point")
    responses = np.asarray([reference_point])

    pref, plot = evolver.start()
    # response = evolver.population.ideal_fitness_val + [0.04, 0.04, 0.4]
    pref.response = pd.DataFrame(
        [responses[0]], columns=pref.content["dimensions_data"].columns
    )

    pref, plot = evolver.iterate(pref)
    i2, obj_values, base  = evolver.end()



    if len(obj_values)>0:
        radius = 0.5
        w_point = reference_point + 2 * np.ones(5)
        #PF, PFsize = pf_samples(int(objectives), no_layers, no_gaps, shrink_factors, igdsamSize, id, radius, reference_point, w_point)
        RNSGA2, RNSGA2_size    = preprocessing_asf(obj_values, reference_point, w_point, radius)
        RNSGA2_IGD   = cal_hv(RNSGA2, RNSGA2_size, w_point)
        #print(RNSGA2_IGD)
    else:
        RNSGA2_IGD = 0

    #print(RNSGA2_IGD) 
    return RNSGA2_IGD

if __name__ == "__main__":
    #problem = river_pollution_problem
    generations = 800
    mu = 0.05
    reference_point = [5.3,2.9,7.2,-1.3,0.25]

    crossover= "SBX"
    crossover_probability = 0.9
    crossover_repair = "bounds"
    crossover_distribution_index = 10
    blx_alpha_crossover = None
    mutation = "polynomial"
    mutation_probability= 0.5
    mutation_repair = "bounds"
    polinomial_mut_dist_index = 20
    uniform_mut_perturbation = None
    selection = "random"
    tournament_size = None




    no_layers = 2                  # number of layers
    no_gaps   = [3, 2]             # specify the # of divisions on each layer
    shrink_factors = [1.0, 0.5]    # shrinkage factor for each layer
    igdsamSize = 10000
    radius = 0.5
    results = []
    for k in range(0,30):
        print("Run ",k)
        result = run(reference_point, generations, None, mu, selection, tournament_size, crossover, crossover_probability, blx_alpha_crossover, crossover_distribution_index, crossover_repair, mutation, mutation_probability, mutation_repair, uniform_mut_perturbation, polinomial_mut_dist_index)
        results.append(result)
    file_name = f'Results_for_River_Pollution_RNSGA3_default'
    np.savetxt(file_name,results)
  