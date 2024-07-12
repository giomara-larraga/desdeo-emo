import numpy as np
import pandas as pd
import copy
import baseADM
from baseADM import *
import generatePreference as gp

from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors

from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.IMOEAD import IMOEA_D as MOEA_D

from pymoo.factory import get_problem, get_reference_directions
import rmetric as rm
from sklearn.preprocessing import Normalizer
#from pymoo.config import Configuration

#Configuration.show_compile_hint = False

problem_names = ["DTLZ2", "DTLZ3","DTLZ4"]
n_objs = np.asarray([4, 7, 9])  # number of objectives

K = 10
n_vars = K + n_objs - 1  # number of variables

num_gen_per_iter = [200]  # number of generations per iteration

algorithms = ["iRVEA", "iMOEA_D"]  # algorithms to be compared

# the followings are for formatting results
column_names = (
    ["problem", "num_obj", "iteration", "num_gens", "preferred_ranges"]
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

total_run = 5
for run in range(total_run):
    print(f"Run {run+1} of {total_run}")
    counter = 1
    total_count = len(num_gen_per_iter) * len(n_objs) * len(problem_names)
    for gen in num_gen_per_iter:
        for n_obj, n_var in zip(n_objs, n_vars):
            for problem_name in problem_names:
                print(f"Loop {counter} of {total_count}")
                counter += 1
                problem = test_problem_builder(
                    name=problem_name, n_of_objectives=n_obj, n_of_variables=n_var
                )

                problem.ideal = np.asarray([0.1e-500] * n_obj)
                problem.nadir = abs(np.random.normal(size=n_obj, scale=0.15)) + 1

                true_ideal = np.asarray([0] * n_obj)
                true_nadir = np.asarray([1] * n_obj)

                # interactive
                int_rvea = RVEA(problem=copy.copy(problem), interact=True, n_gen_per_iter=gen)
                int_nsga = MOEA_D(problem=copy.copy(problem), interact=True, n_gen_per_iter=gen)

                # initial reference point is specified randomly
                response = np.random.rand(n_obj)

                # run algorithms once with the randomly generated reference point
                pref_int_rvea,_ = int_rvea.requests()
                pref_int_nsga,_ = int_nsga.requests()
                pref_int_rvea[2].response = pd.DataFrame(
                    [response],
                    columns=pref_int_rvea[2].content["dimensions_data"].columns,
                )
                pref_int_nsga[2].response = pd.DataFrame(
                    [response],
                    columns=pref_int_nsga[2].content["dimensions_data"].columns,
                )

                pref_int_rvea,_ = int_rvea.iterate(pref_int_rvea[2])
                pref_int_nsga,_ = int_nsga.iterate(pref_int_nsga[2])

                # build initial composite front
                (
                    rvea_n_solutions,
                    nsga_n_solutions,
                    cf,
                ) = generate_composite_front_with_identity(
                    int_rvea.population.objectives, int_nsga.population.objectives
                )

                # the following two lines for getting pareto front by using pymoo framework
                problemR = get_problem(problem_name.lower(), n_var, n_obj)
                ref_dirs = get_reference_directions(
                    "das-dennis", n_obj, n_partitions=12
                )
                pareto_front = problemR.pareto_front(ref_dirs)

                # creates uniformly distributed reference vectors
                reference_vectors = ReferenceVectors(lattice_resolution, n_obj)

                # learning phase
                for i in range(L):
                    data_row[["problem", "num_obj", "iteration", "num_gens"]] = [
                        problem_name,
                        n_obj,
                        i + 1,
                        gen,
                    ]

                    problem.ideal = np.asarray([0.1e-500] * n_obj)
                    problem.nadir = abs(np.random.normal(size=n_obj, scale=0.15)) + 1

                    # After this class call, solutions inside the composite front are assigned to reference vectors
                    base = baseADM(cf, reference_vectors)

                    # generates the next reference point for the next iteration in the learning phase
                    response,reference_point = gp.generateRanges4learning(base, problem.ideal, true_nadir)

                    data_row["preferred_ranges"] = [response]
                    print(response)
      
                    print(problem.ideal)
                    # run algorithms with the new reference point
                    pref_int_rvea[3].response = response
                    pref_int_nsga[3].response = response
                    previous_RVEA_FEs = int_rvea._function_evaluation_count
                    previous_NSGA_FEs = int_nsga._function_evaluation_count
                    pref_int_rvea,_ = int_rvea.iterate(pref_int_rvea[3])
                    pref_int_nsga,_ = int_nsga.iterate(pref_int_nsga[3])

                    peritr_RVEA_FEs = (
                        int_rvea._function_evaluation_count - previous_RVEA_FEs
                    )
                    peritr_NSGA_FEs = (
                        int_nsga._function_evaluation_count - previous_NSGA_FEs
                    )

                    # extend composite front with newly obtained solutions
                    (
                        rvea_n_solutions,
                        nsga_n_solutions,
                        cf,
                    ) = generate_composite_front_with_identity(
                        int_rvea.population.objectives,
                        int_nsga.population.objectives,
                        cf,
                    )

                    data_row["iRVEA_N_Ss"] = [rvea_n_solutions]
                    data_row["iMOEA_D_N_Ss"] = [nsga_n_solutions]
                    data_row["iRVEA_FEs"] = [peritr_RVEA_FEs * n_obj]
                    data_row["iMOEA_D_FEs"] = [peritr_NSGA_FEs * n_obj]

                    # R-metric calculation
                    ref_point = reference_point.reshape(1, n_obj)

                    # normalize reference point
                    rp_transformer = Normalizer().fit(ref_point)
                    norm_rp = rp_transformer.transform(ref_point)

                    rmetric = rm.RMetric(problemR, norm_rp, pf=pareto_front)

                    # normalize solutions before sending r-metric
                    rvea_transformer = Normalizer().fit(int_rvea.population.objectives)
                    norm_rvea = rvea_transformer.transform(
                        int_rvea.population.objectives
                    )

                    nsga_transformer = Normalizer().fit(int_nsga.population.objectives)
                    norm_nsga = nsga_transformer.transform(
                        int_nsga.population.objectives
                    )

                    # R-metric calls for R_IGD and R_HV
                    rigd_irvea, rhv_irvea = rmetric.calc(norm_rvea, others=norm_nsga)
                    rigd_insga, rhv_insga = rmetric.calc(norm_nsga, others=norm_rvea)

                    data_row[
                        ["iRVEA" + excess_col for excess_col in excess_columns]
                    ] = [rigd_irvea, rhv_irvea]
                    data_row[
                        ["iMOEA_D" + excess_col for excess_col in excess_columns]
                    ] = [rigd_insga, rhv_insga]

                    data = data.append(data_row, ignore_index=1)

                # Decision phase
                # After the learning phase the reference vector which has the maximum number of assigned solutions forms ROI
                max_assigned_vector = gp.get_max_assigned_vector(base.assigned_vectors)

                for i in range(D):
                    data_row[["problem", "num_obj", "iteration", "num_gens"]] = [
                        problem_name,
                        n_obj,
                        L + i + 1,
                        gen,
                    ]

                    problem.ideal = np.asarray([0.1e-500] * n_obj)
                    problem.nadir = abs(np.random.normal(size=n_obj, scale=0.15)) + 1

                    # since composite front grows after each iteration this call should be done for each iteration
                    base = baseADM(cf, reference_vectors)

                    # generates the next reference point for the decision phase
                    response, reference_point = gp.generateRanges4decision(
                        base, max_assigned_vector[0], problem.ideal, true_nadir
                    )

                    data_row["preferred_ranges"] = [response]

                    # run algorithms with the new reference point
                    pref_int_rvea[3].response = response
                    pref_int_nsga[3].response = response

                    previous_RVEA_FEs = int_rvea._function_evaluation_count
                    previous_NSGA_FEs = int_nsga._function_evaluation_count
                    pref_int_rvea,_ = int_rvea.iterate(pref_int_rvea[3])
                    pref_int_nsga,_ = int_nsga.iterate(pref_int_nsga[3])

                    peritr_RVEA_FEs = (
                        int_rvea._function_evaluation_count - previous_RVEA_FEs
                    )
                    peritr_NSGA_FEs = (
                        int_nsga._function_evaluation_count - previous_NSGA_FEs
                    )
                    # extend composite front with newly obtained solutions
                    (
                        rvea_n_solutions,
                        nsga_n_solutions,
                        cf,
                    ) = generate_composite_front_with_identity(
                        int_rvea.population.objectives,
                        int_nsga.population.objectives,
                        cf,
                    )

                    data_row["iRVEA_N_Ss"] = [rvea_n_solutions]
                    data_row["iMOEA_D_N_Ss"] = [nsga_n_solutions]
                    data_row["iRVEA_FEs"] = [peritr_RVEA_FEs * n_obj]
                    data_row["iMOEA_D_FEs"] = [peritr_NSGA_FEs * n_obj]

                    # R-metric calculation
                    ref_point = reference_point.reshape(1, n_obj)

                    rp_transformer = Normalizer().fit(ref_point)
                    norm_rp = rp_transformer.transform(ref_point)

                    # for decision phase, delta is specified as 0.2
                    rmetric = rm.RMetric(problemR, norm_rp, delta=0.2, pf=pareto_front)

                    # normalize solutions before sending r-metric
                    rvea_transformer = Normalizer().fit(int_rvea.population.objectives)
                    norm_rvea = rvea_transformer.transform(
                        int_rvea.population.objectives
                    )

                    nsga_transformer = Normalizer().fit(int_nsga.population.objectives)
                    norm_nsga = nsga_transformer.transform(
                        int_nsga.population.objectives
                    )

                    rigd_irvea, rhv_irvea = rmetric.calc(norm_rvea, others=norm_nsga)
                    rigd_insga, rhv_insga = rmetric.calc(norm_nsga, others=norm_rvea)

                    data_row[
                        ["iRVEA" + excess_col for excess_col in excess_columns]
                    ] = [rigd_irvea, rhv_irvea]
                    data_row[
                        ["iMOEA_D" + excess_col for excess_col in excess_columns]
                    ] = [rigd_insga, rhv_insga]

                    data = data.append(data_row, ignore_index=1)

    data.to_csv(f"./results/ranges{run+1}.csv", index=False)
