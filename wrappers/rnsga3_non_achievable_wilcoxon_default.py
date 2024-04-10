import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.EAs.RNSGAIII import RNSGAIII
from desdeo_emo.utilities.pf_samples import pf_samples
from desdeo_emo.utilities.preprocessing_asf import preprocessing_asf
from desdeo_emo.utilities.cal_metric import cal_metric

import csv
from ast import literal_eval

import csv



def read_csv_with_arrays(file_path):
    data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        header_skipped = False

        for row in reader:
            if not header_skipped:
                header_skipped = True
                continue  # Skip the header row
            row_data = []
            for index, item in enumerate(row):
                if index == 4:  # Check if it's the refpoinr column
                    # Split the string by comma and try converting each element to float
                    refpoinr_values = []
                    for value in item.split(','):
                        value = value.strip()
                        try:
                            refpoinr_values.append(float(value))
                        except ValueError:
                            print(f"Warning: Skipping non-numeric value '{value}' in refpoinr column.")
                    row_data.append(refpoinr_values)
                else:
                    row_data.append(item)
            data.append(row_data)
    return data



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
    #problems = np.array(["DTLZ2","DTLZ3", "DTLZ4"])
    #ids = np.array([5,5,5])
    #objectives = np.array([3,5,7,9])
    #variables = np.array([12,14,16,18])
    #generations = np.array([400,600,800,1000])
    #mu = np.array([0.1, 0.05, 0.05, 0.05])
    #reference_points = np.array([[0.7,0.8,0.5],[0.7,0.7,0.8,0.9,0.5], [0.7,0.5,0.7,0.8,0.9,0.6,0.5], [0.7,0.4,0.5,0.7,0.8,0.5,0.9,0.6,0.5]])
    #reference_points = np.array([[0.2,0.5,0.6],[0.2,0.5,0.5,0.2,0.6],[0.2,0.5,0.5,0.2,0.2,0.5,0.6],[0.2,0.5,0.2,0.2,0.2,0.2,0.2,0.5,0.6]])

    crossover= np.array(["SBX", "SBX", "SBX", "SBX"])
    crossover_probability = np.array([0.9, 0.9, 0.9, 0.9])
    crossover_repair = np.array(["bounds", "bounds", "bounds", "bounds"])
    crossover_distribution_index = np.array([10, 10, 10, 10])
    blx_alpha_crossover = np.array([None, None, None,None])
    mutation = np.array(["polynomial", "polynomial", "polynomial", "polynomial"])
    mutation_probability= np.array([1/12,1/14,1/16,1/18])
    mutation_repair = np.array(["bounds","bounds","bounds","bounds"])
    polinomial_mut_dist_index = np.array([20, 20, 20,20])
    uniform_mut_perturbation = np.array([None, None, None, None])
    selection = np.array(["random","random","random","random"])
    tournament_size = np.array([None,None,None,None])

    file_path = '/home/giomara/WCCI2024/desdeo-emo/wrappers/data_default_parameters_non_dominated.csv'
    csv_data = read_csv_with_arrays(file_path)
    #Print the data
    for row in csv_data[1:]:
        problem_name= row[0]
        id_prob= int(row[1]) 
        objectives= int(row[2])
        variables= int(row[3])
        ref_point= row[4]
        generations= int(row[5])
        mu= float(row[6])
        #print(problem)

        print("Starting ", problem_name, "objectives ", objectives)
        problem = test_problem_builder(name=problem_name, n_of_variables=variables, n_of_objectives= objectives)

        # only useful for the many-objective scenario (i.e., objDim > 3)
        no_layers = 2                  # number of layers
        no_gaps   = [3, 2]             # specify the # of divisions on each layer
        shrink_factors = [1.0, 0.5]    # shrinkage factor for each layer
        igdsamSize = 10000
        # set trimming radius
        if objectives < 5:
            radius = 0.2
        else:
            radius = 0.5
        results = []

        if objectives == 3:
            mutation_probability= np.array([1/12])
        elif objectives == 5:
            mutation_probability= np.array([1/14])
        else:
            mutation_probability= np.array([1/16])

        for k in range(0,10):
            #print(problems[i], "run ", k)
            result = run(problem, id_prob, objectives, ref_point, generations, None, mu, selection[0], tournament_size[0], crossover[0], crossover_probability[0], blx_alpha_crossover[0], crossover_distribution_index[0], crossover_repair[0], mutation[0], mutation_probability, mutation_repair[0], uniform_mut_perturbation[0], polinomial_mut_dist_index[0])
            results.append(result)
        file_name = f'Results_default_for_{problem_name}_{objectives}_non_achievable'
        np.savetxt(file_name,results)
        #print (results)

        #print(row[4])
# 

'''
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
            for k in range(0,30):
                #print(problems[i], "run ", k)
                result = run(problem, ids[i], objectives[j], reference_points[j], generations[j], None, mu[j], selection[j], tournament_size[j], crossover[j], crossover_probability[j], blx_alpha_crossover[j], crossover_distribution_index[j], crossover_repair[j], mutation[j], mutation_probability[j], mutation_repair[j], uniform_mut_perturbation[j], polinomial_mut_dist_index[j])
                results.append(result)

            file_name = f'Results_default_for_{problems[i]}_{objectives[j]}_non_achievable'
            np.savetxt(file_name,results)
            #print (results)
'''