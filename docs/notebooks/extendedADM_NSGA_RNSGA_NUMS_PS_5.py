import numpy as np
import pandas as pd

import baseADM
from baseADM import *
import generatePreference as gp

from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors

from desdeo_emo.EAs.RVEA import RVEA as NSGAIII
from desdeo_emo.EAs.NSGAIIINUMS import NSGAIIINUMS as NUMS
from desdeo_emo.EAs.RNSGAIII import RNSGAIII


from desdeo_emo.utilities.preference_converters import UPEMO

from pymoo.factory import get_problem, get_reference_directions
import rmetric as rm
from sklearn.preprocessing import Normalizer
#from pymoo.config import Configuration

#Configuration.show_compile_hint = False

problem_names = ["DTLZ1", "DTLZ3"]
#problem_names = ["DTLZ1"]
n_objs = np.asarray([5])  # number of objectives

K = 10
n_vars = K + n_objs - 1  # number of variables

num_gen_per_iter = [200]  # number of generations per iteration

algorithms = ["iNSGAIII", "NUMS", "RNSGAIII"]  # algorithms to be compared

# the followings are for formatting results
column_names = (
    ["problem", "num_obj", "iteration", "num_gens", "reference_point"]
    + [algorithm + "_R_IGD" for algorithm in algorithms]
    + [algorithm + "_R_HV" for algorithm in algorithms]
    + [algorithm + "_N_Ss" for algorithm in algorithms]
    + [algorithm + "_FEs" for algorithm in algorithms]
)
excess_columns = ["_R_IGD", "_R_HV"]
data = pd.DataFrame(columns=column_names)
data_row = pd.DataFrame(columns=column_names, index=[1])

# ADM parameters
L = 4  # number of iterations for the learning phase
D = 3  # number of iterations for the decision phase
lattice_resolution = 5  # density variable for creating reference vectors

total_run = 10

for gen in num_gen_per_iter:
    for n_obj, n_var in zip(n_objs, n_vars):
        for problem_name in problem_names:
            #Global arrays for median and stddev
            
            median_ns_nsga_learning = []
            median_rm_nsga_learning = []
            median_ns_nums_learning = []
            median_rm_nums_learning = []
            median_ns_rnsga_learning = []
            median_rm_rnsga_learning = []

            median_ns_nsga_decision = []
            median_rm_nsga_decision = []
            median_ns_nums_decision = []
            median_rm_nums_decision = []
            median_ns_rnsga_decision = []
            median_rm_rnsga_decision = []

            for run in range(total_run):
                print(f"Problem {problem_name} Objectives {n_obj} Run {run+1} of {total_run}")
              
                problem = test_problem_builder(
                    name=problem_name, n_of_objectives=n_obj, n_of_variables=n_var
                )
                problem.ideal_fitness = np.asarray([0] * n_obj)
                problem.nadir_fitness = abs(np.random.normal(size=n_obj, scale=0.15)) + 1
                true_nadir = np.asarray([1] * n_obj)
                # initial reference point is specified randomly
                response = np.random.rand(n_obj)
                # run algorithms once with the randomly generated reference point
                # interactive RVEA
                
                int_nsga = NSGAIII(problem=problem, interact=True, n_gen_per_iter=gen)
                int_nums = NUMS(problem=problem, interact=True, n_gen_per_iter=gen)
                int_rnsga = RNSGAIII(problem=problem, interact=True, n_gen_per_iter=gen)

                int_nsga.set_interaction_type('Reference point')
                int_nums.set_interaction_type('Reference point')
                int_rnsga.set_interaction_type('Reference point')

                pref_int_nsga, _ = int_nsga.start()
                pref_int_nums, _ = int_nums.start()
                pref_int_rnsga, _ = int_rnsga.start()

                for boundrnd in range(0, n_obj):
                    if response[boundrnd] < int_nsga.population.problem.ideal_fitness[boundrnd]:
                        response[boundrnd] = int_nsga.population.problem.nadir_fitness[boundrnd]
                #print(pref_int_rvea.content["dimensions_data"].columns["ideal"])

                pref_int_nsga.response = pd.DataFrame(
                    [response],
                    columns=pref_int_nsga.content["dimensions_data"].columns,
                )

                pref_int_nums.response = pd.DataFrame(
                    [response],
                    columns=pref_int_nums.content["dimensions_data"].columns,
                )

                pref_int_rnsga.response = pd.DataFrame(
                    [response],
                    columns=pref_int_rnsga.content["dimensions_data"].columns,
                )
               
                pref_int_nsga,_ = int_nsga.iterate(pref_int_nsga)
                pref_int_nums,_ = int_nums.iterate(pref_int_nums)
                pref_int_rnsga,_ = int_rnsga.iterate(pref_int_rnsga)

                # build initial composite front
                (
                    nsga_n_solutions,
                    nums_n_solutions,
                    rnsga_n_solutions,
                    cf,
                ) = generate_composite_front_with_identity_2(
                    int_nsga.population.objectives, int_nums.population.objectives, int_rnsga.population.objectives
                )

                # the following two lines for getting pareto front by using pymoo framework
                problemR = get_problem(problem_name.lower(), n_var, n_obj)
                ref_dirs = get_reference_directions(
                    "das-dennis", n_obj, n_partitions=12
                )
                pareto_front = problemR.pareto_front(ref_dirs)
                # creates uniformly distributed reference vectors
                reference_vectors = ReferenceVectors(lattice_resolution=lattice_resolution, number_of_objectives=n_obj)

                # Arrays to store values for the current run
                rm_nsga_learning = np.array([])    
                rm_nsga_decision = np.array([])
                ns_nsga_learning = np.array([]) 
                ns_nsga_decision = np.array([]) 

                rm_nums_learning = np.array([])    
                rm_nums_decision = np.array([])
                ns_nums_learning = np.array([]) 
                ns_nums_decision = np.array([]) 

                rm_rnsga_learning = np.array([])    
                rm_rnsga_decision = np.array([])
                ns_rnsga_learning = np.array([]) 
                ns_rnsga_decision = np.array([]) 


                # learning phase
                for i in range(L):
                    data_row[["problem", "num_obj", "iteration", "num_gens"]] = [
                        problem_name,
                        n_obj,
                        i + 1,
                        gen,
                    ]
                    # After this class call, solutions inside the composite front are assigned to reference vectors
                    base = baseADM(cf, reference_vectors)
                    # generates the next reference point for the next iteration in the learning phase
                    response = gp.preferredSolutions4learning(base)
                    #print(response)
                    #response, reference_point = gp.generateRanges4learning(base, problem.ideal, true_nadir)
                    int_nsga.set_interaction_type('Preferred solutions')
                    int_nums.set_interaction_type('Reference point')
                    int_rnsga.set_interaction_type('Reference point')

                    pref_int_nsga, _ = int_nsga.start()
                    pref_int_nums, _ = int_nums.start()
                    pref_int_rnsga, _ = int_rnsga.start()

                    #data_row["reference_point"] = [response]
                    # run algorithms with the new reference point
                    # Interactive RVEA
                    pref_int_nsga.response = response[0]
                    pref_int_nums.response = pd.DataFrame(
                    response,
                    columns=pref_int_nums.content["dimensions_data"].columns,
                    )

                    pref_int_rnsga.response = pd.DataFrame(
                        response,
                        columns=pref_int_rnsga.content["dimensions_data"].columns,
                    )
                   
                    previous_NSGA_FEs = int_nsga._function_evaluation_count
                    previous_NUMS_FEs = int_nums._function_evaluation_count
                    previous_RNSGA_FEs = int_rnsga._function_evaluation_count

                    pref_int_nsga,_ = int_nsga.iterate(pref_int_nsga)
                    pref_int_nums,_ = int_nums.iterate(pref_int_nums)
                    pref_int_rnsga,_ = int_rnsga.iterate(pref_int_rnsga)

                    peritr_NSGA_FEs = (
                        int_nsga._function_evaluation_count - previous_NSGA_FEs
                    )

                    peritr_NUMS_FEs = (
                        int_nums._function_evaluation_count - previous_NUMS_FEs
                    )
                    peritr_RNSGA_FEs = (
                        int_rnsga._function_evaluation_count - previous_RNSGA_FEs
                    )
                    # extend composite front with newly obtained solutions
                    (
                        nsga_n_solutions,
                        nums_n_solutions,
                        rnsga_n_solutions,
                        cf,
                    ) = generate_composite_front_with_identity_2(
                        int_nsga.population.objectives,
                        int_nums.population.objectives,
                        int_rnsga.population.objectives,
                        cf,
                    )
                    #data_row["iRVEA_N_Ss"] = [rvea_n_solutions]
                    #data_row["RNSGAIII_N_Ss"] = [nsga_n_solutions]
                    #data_row["iRVEA_FEs"] = [peritr_RVEA_FEs * n_obj]
                    #data_row["RNSGAIII_FEs"] = [peritr_NSGA_FEs * n_obj]

                    
                    ns_nsga_learning = np.append(ns_nsga_learning, nsga_n_solutions)
                    ns_nums_learning = np.append(ns_nums_learning, nums_n_solutions)
                    ns_rnsga_learning = np.append(ns_rnsga_learning, rnsga_n_solutions)


                # Compute cumulative sum of the learning phase
                median_ns_nsga_learning = np.append(median_ns_nsga_learning, np.sum(ns_nsga_learning))
                median_ns_nums_learning = np.append(median_ns_nums_learning, np.sum(ns_nums_learning))
                median_ns_rnsga_learning = np.append(median_ns_rnsga_learning, np.sum(ns_rnsga_learning))
            
                # Decision phase
                # After the learning phase the reference vector which has the maximum number of assigned solutions forms ROI
                max_assigned_vector = gp.get_max_assigned_vector(base.assigned_vectors)
 
                for i in range(D):
                    #print("Decision phase")
                    data_row[["problem", "num_obj", "iteration", "num_gens"]] = [
                        problem_name,
                        n_obj,
                        L + i + 1,
                        gen,
                    ]
                    # since composite front grows after each iteration this call should be done for each iteration
                    base = baseADM(cf, reference_vectors)
                    response = gp.preferredSolutions4Decision2(base)
                    #print(response)
                    #response, reference_point = gp.generateRanges4learning(base, problem.ideal, true_nadir)
                    #int_nsga.set_interaction_type('Preferred solutions')
                    int_nums.set_interaction_type('Preferred solutions')
                    int_rnsga.set_interaction_type('Preferred solutions')

                    pref_int_nsga, _ = int_nsga.start()
                    pref_int_nums, _ = int_nums.start()
                    pref_int_rnsga, _ = int_rnsga.start()

                    #data_row["reference_point"] = [response]
                    # run algorithms with the new reference point
                    # Interactive RVEA
                    pref_int_nsga.response = response
                    pref_int_nums.response = response

                    pref_int_rnsga.response = response

                    previous_NSGA_FEs = int_nsga._function_evaluation_count
                    previous_NUMS_FEs = int_nums._function_evaluation_count
                    previous_RNSGA_FEs = int_rnsga._function_evaluation_count

                    pref_int_nsga,_ = int_nsga.iterate(pref_int_nsga)
                    pref_int_nums,_ = int_nums.iterate(pref_int_nums)
                    pref_int_rnsga,_ = int_rnsga.iterate(pref_int_rnsga)

                    peritr_NSGA_FEs = (
                        int_nsga._function_evaluation_count - previous_NSGA_FEs
                    )

                    peritr_NUMS_FEs = (
                        int_nums._function_evaluation_count - previous_NUMS_FEs
                    )
                    peritr_RNSGA_FEs = (
                        int_rnsga._function_evaluation_count - previous_RNSGA_FEs
                    )
                    # extend composite front with newly obtained solutions
                    (
                        nsga_n_solutions,
                        nums_n_solutions,
                        rnsga_n_solutions,
                        cf,
                    ) = generate_composite_front_with_identity_2(
                        int_nsga.population.objectives,
                        int_nums.population.objectives,
                        int_rnsga.population.objectives,
                        cf,
                    )
                    ns_nsga_decision= np.append(ns_nsga_decision, nsga_n_solutions)
                    ns_nums_decision= np.append(ns_nums_decision, nums_n_solutions)
                    ns_rnsga_decision= np.append(ns_rnsga_decision, rnsga_n_solutions)
                #Compute median of the decision phase
                median_ns_nsga_decision = np.append(median_ns_nsga_decision, np.sum(ns_nsga_decision))
                median_ns_nums_decision = np.append(median_ns_nums_decision, np.sum(ns_nums_decision))
                median_ns_rnsga_decision = np.append(median_ns_rnsga_decision, np.sum(ns_rnsga_decision))

            print(f"Results for Problem {problem_name} with {n_obj} objectives")
            print ("Learning phase")
            print("Ns")
            print (f"{np.mean(median_ns_nsga_learning):.1f} & {np.std(median_ns_nsga_learning):.4f} & {np.mean(median_ns_nums_learning):.1f} & {np.std(median_ns_nums_learning):.4f} & {np.mean(median_ns_rnsga_learning):.1f} & {np.std(median_ns_rnsga_learning):.4f}")
    

            print("Decision phase")
            print("Ns")
            print (f"{np.mean(median_ns_nsga_decision):.1f} & {np.std(median_ns_nsga_decision):.4f} & {np.mean(median_ns_nums_decision):.1f} & {np.std(median_ns_nums_decision):.4f} & {np.mean(median_ns_rnsga_decision):.1f} & {np.std(median_ns_rnsga_decision):.4f}")
        

    #data.to_csv(f"refpoints{run+1}.csv", index=False)
