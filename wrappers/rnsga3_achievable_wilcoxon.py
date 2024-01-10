import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from desdeo_problem import MOProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.EAs.RNSGAIII import RNSGAIII
from desdeo_emo.utilities.pf_samples import pf_samples
from desdeo_emo.utilities.preprocessing_asf import preprocessing_asf
from desdeo_emo.utilities.cal_metric import cal_metric

#problem_name = "ZDT1"
#problem = test_problem_builder(problem_name)

def run(problem, id, objectives, reference_point, generations, population_size, mu, selection, tournament_size, crossover, crossover_probability, blx_alpha_crossover, crossover_distribution_index, crossover_repair, mutation,mutation_probability, mutation_repair, uniform_mut_perturbation, polinomial_mut_dist_index):
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
        w_point = reference_point + 2 * np.ones(objectives)
        PF, PFsize = pf_samples(int(objectives), no_layers, no_gaps, shrink_factors, igdsamSize, id, radius, reference_point, w_point)
        RNSGA2, RNSGA2_size    = preprocessing_asf(obj_values, reference_point, w_point, radius)
        RNSGA2_IGD   = cal_metric(RNSGA2, PF, w_point, RNSGA2_size, PFsize)
        #print(RNSGA2_IGD)
    else:
        RNSGA2_IGD = 0

    #print(RNSGA2_IGD) 
    return RNSGA2_IGD

if __name__ == "__main__":
    problems = np.array(["DTLZ2", "DTLZ4"])
    ids = np.array([5,5])
    objectives = np.array([3,5,7,9])
    variables = np.array([12,14,16,18])
    generations = np.array([400,600,800,1000])
    mu = np.array([0.1, 0.05, 0.05, 0.05])
    reference_points = np.array([[0.7,0.8,0.5],[0.7,0.7,0.8,0.9,0.5], [0.7,0.5,0.7,0.8,0.9,0.6,0.5], [0.7,0.4,0.5,0.7,0.8,0.5,0.9,0.6,0.5]])
    #non_achievable_reference_points = np.array([[0.2,0.5,0.6],[0.2,0.5,0.5,0.2,0.6],[0.2,0.5,0.5,0.2,0.2,0.5,0.6],[0.2,0.5,0.2,0.2,0.2,0.2,0.2,0.5,0.6]])

    crossover= np.array(["SBX", "SBX", "SBX", "SBX"])
    crossover_probability = np.array([0.4413, 0.0798, 0.0741, 0.3503])
    crossover_repair = np.array(["random", "bounds", "round", "round"])
    crossover_distribution_index = np.array([68.8883, 256.1593, 136.7326, 261.9070])
    blx_alpha_crossover = np.array([None, None, None,None])
    mutation = np.array(["polynomial", "uniform", "uniform", "uniform"])
    mutation_probability= np.array([0.0892,0.2520,0.0205,0.2352])
    mutation_repair = np.array(["round","round","round","bounds"])
    polinomial_mut_dist_index = np.array([51.3330, None, None,None])
    uniform_mut_perturbation = np.array([None, 0.5163,0.5571,0.8753])
    selection = np.array(["tournament","tournament","tournament","tournament"])
    tournament_size = np.array([4,7,3,5])


    for i in range(len(problems)):
        for j in range(len(objectives)):
            print("Starting ", problems[i], "objectives ", objectives[j])
            problem = test_problem_builder(name=problems[i], n_of_variables=variables[j], n_of_objectives= objectives[j])
            # only useful for the many-objective scenario (i.e., objDim > 3)
            no_layers = 2                  # number of layers
            no_gaps   = [3, 2]             # specify the # of divisions on each layer
            shrink_factors = [1.0, 0.5]    # shrinkage factor for each layer
            igdsamSize = 10000
            # set trimming radius
            if objectives[j] < 5:
                radius = 0.2
            else:
                radius = 0.5

            results = []
            for k in range(0,15):
                #print(problems[i], "run ", k)
                result = run(problem, ids[i], objectives[j], reference_points[j], generations[j], None, mu[j], selection[j], tournament_size[j], crossover[j], crossover_probability[j], blx_alpha_crossover[j], crossover_distribution_index[j], crossover_repair[j], mutation[j], mutation_probability[j], mutation_repair[j], uniform_mut_perturbation[j], polinomial_mut_dist_index[j])
                results.append(result)

            file_name = f'Results_for_{problems[i]}_{objectives[j]}_achievable'
            np.savetxt(file_name,results)
            #print (results)