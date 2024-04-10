import numpy as np
import pandas as pd
import pygmo

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from desdeo_problem import MOProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.EAs.RNSGAII import RNSGAII
from desdeo_emo.EAs.MOEAD import MOEA_D
from desdeo_emo.utilities.pf_samples import pf_samples
from desdeo_emo.utilities.preprocessing_asf import preprocessing_asf
from desdeo_emo.utilities.cal_metric import cal_metric
from desdeo_emo.utilities.cal_metric import cal_hv
from desdeo_problem.testproblems.RiverPollution import river_pollution_problem
from desdeo_problem import variable_builder, ScalarObjective, VectorObjective, MOProblem
from desdeo_problem.testproblems.VehicleCrashworthiness import vehicle_crashworthiness


def run(reference_point, generations, population_size, mu, selection, tournament_size, crossover, crossover_probability, blx_alpha_crossover, crossover_distribution_index, crossover_repair, mutation,mutation_probability, mutation_repair, uniform_mut_perturbation, polinomial_mut_dist_index):
    problem = vehicle_crashworthiness()
    evolver = RNSGAII(
        problem,
        n_iterations=1,
        n_gen_per_iter=generations,
        population_size=300,
        interact=True,
        epsilon=0.01,
        normalization="front",
        weights=None,
        extreme_points_as_reference_points=False,
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
    i2, obj_values = evolver.end()

    #df = pd.DataFrame(obj_values, columns=['Variable1', 'Variable2', 'Variable3', 'v4', 'v5'])
    #
    ## Create a parallel coordinates plot
    #fig = go.Figure(data=go.Parcoords(
    #    line=dict(color=df.index, colorscale='Viridis', showscale=True),
    #    dimensions=[dict(label=col, values=df[col]) for col in df.columns]
    #))
    #
    ## Customize the plot layout
    #fig.update_layout(
    #    title='Parallel Coordinates Plot',
    #    xaxis=dict(showline=True, showgrid=False),
    #    yaxis=dict(showline=True, showgrid=False),
    #    showlegend=False
    #)
    #
    #fig.show()

    if len(obj_values)>0:
        radius = 0.5
        w_point = reference_point + 2 * np.ones(3)
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
    generations = 500
    mu = 0.01
    reference_point = [1662.233, 8.150, 0.082]
    
    crossover="SBX"
    crossover_probability = 0.1001
    crossover_repair = "round"
    crossover_distribution_index = 153.4679
    blx_alpha_crossover = None
    mutation = "uniform"
    mutation_probability= 0.2636
    mutation_repair = "bounds"
    polinomial_mut_dist_index = None
    uniform_mut_perturbation = 0.5224
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
        #print(result)
    file_name = f'Results_for_Car_RNSGA2_opt'
    np.savetxt(file_name,results)
  