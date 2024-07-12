import numpy as np
import pandas as pd

import baseADM
from baseADM import *
import generatePreference as gp

from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors

from desdeo_emo.EAs.RVEA import RVEA

# from desdeo_emo.EAs.NSGAIII import NSGAIII

from pymoo.factory import get_problem, get_reference_directions
import rmetric as rm
from sklearn.preprocessing import Normalizer
from pymoo.configuration import Configuration

Configuration.show_compile_hint = False

problem_names = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4"]
n_objs = np.asarray([3, 4, 5, 6, 7, 8, 9])  # number of objectives

K = 10
n_vars = K + n_objs - 1  # number of variables

num_gen_per_iter = [100]  # number of generations per iteration

algorithms = ["iRVEA_RP", "iRVEA_Ranges"]  # algorithms to be compared

# the followings are for formatting results
column_names = (
    [
        "problem",
        "num_obj",
        "iteration",
        "num_gens",
        "reference_point",
        "preferred_ranges",
    ]
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

total_run = 2
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

                ideal = np.asarray([0] * n_obj)
                problem.update_ideal(objective_vectors=ideal, fitness=ideal)
                problem.nadir = abs(np.random.normal(size=n_obj, scale=0.15)) + 1

                true_nadir = np.asarray([1] * n_obj)

                # interactive

                int_rvea = RVEA(problem=problem, interact=True, n_gen_per_iter=gen)

                int_rvea_ranges = RVEA(
                    problem=problem, interact=True, n_gen_per_iter=gen
                )

                # initial reference point is specified randomly
                ref_point = np.random.rand(n_obj)

                # run algorithms once with the randomly generated reference point
                _, pref_int_rvea = int_rvea.requests()
                _, pref_int_rvea_ranges = int_rvea_ranges.requests()

                pref_int_rvea[2].response = pd.DataFrame(
                    [ref_point],
                    columns=pref_int_rvea[2].content["dimensions_data"].columns,
                )

                pref_int_rvea_ranges[2].response = pd.DataFrame(
                    [ref_point],
                    columns=pref_int_rvea_ranges[2].content["dimensions_data"].columns,
                )

                _, pref_int_rvea = int_rvea.iterate(pref_int_rvea[2])
                # First run of the preferred ranges based RVEA is also made with the randomly generated reference point
                _, pref_int_rvea_ranges = int_rvea_ranges.iterate(
                    pref_int_rvea_ranges[2]
                )

                # build initial composite front
                (
                    rvea_n_solutions,
                    rvea_ranges_n_solutions,
                    cf,
                ) = generate_composite_front_with_identity(
                    int_rvea.population.objectives,
                    int_rvea_ranges.population.objectives,
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
                    # After this class call, solutions inside the composite front are assigned to reference vectors
                    base = baseADM(cf, reference_vectors)
                    # generates the next reference point for the next iteration in the learning phase
                    ranges, reference_point = gp.generateRanges4learning(
                        base, problem.ideal, problem.nadir
                    )

                    data_row["reference_point"] = [reference_point]
                    data_row["preferred_ranges"] = [ranges]

                    # run algorithms with the new reference point
                    pref_int_rvea[2].response = pd.DataFrame(
                        [reference_point],
                        columns=pref_int_rvea[2].content["dimensions_data"].columns,
                    )
                    pref_int_rvea_ranges[3].response = ranges

                    previous_RVEA_FEs = int_rvea._function_evaluation_count
                    previous_RVEA_Ranges_FEs = (
                        int_rvea_ranges._function_evaluation_count
                    )
                    _, pref_int_rvea = int_rvea.iterate(pref_int_rvea[2])
                    _, pref_int_rvea_ranges = int_rvea_ranges.iterate(
                        pref_int_rvea_ranges[3]
                    )

                    peritr_RVEA_FEs = (
                        int_rvea._function_evaluation_count - previous_RVEA_FEs
                    )
                    peritr_RVEA_Ranges_FEs = (
                        int_rvea_ranges._function_evaluation_count
                        - previous_RVEA_Ranges_FEs
                    )

                    # extend composite front with newly obtained solutions
                    (
                        rvea_n_solutions,
                        rvea_ranges_n_solutions,
                        cf,
                    ) = generate_composite_front_with_identity(
                        int_rvea.population.objectives,
                        int_rvea_ranges.population.objectives,
                        cf,
                    )

                    data_row["iRVEA_RP_N_Ss"] = [rvea_n_solutions]
                    data_row["iRVEA_Ranges_N_Ss"] = [rvea_ranges_n_solutions]
                    data_row["iRVEA_RP_FEs"] = [peritr_RVEA_FEs * n_obj]
                    data_row["iRVEA_Ranges_FEs"] = [peritr_RVEA_Ranges_FEs * n_obj]

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

                    rvea_ranges_transformer = Normalizer().fit(
                        int_rvea_ranges.population.objectives
                    )
                    norm_rvea_ranges = rvea_ranges_transformer.transform(
                        int_rvea_ranges.population.objectives
                    )

                    # R-metric calls for R_IGD and R_HV
                    rigd_irvea, rhv_irvea = rmetric.calc(
                        norm_rvea, others=norm_rvea_ranges
                    )
                    rigd_irvea_ranges, rhv_irvea_ranges = rmetric.calc(
                        norm_rvea_ranges, others=norm_rvea
                    )

                    data_row[
                        ["iRVEA_RP" + excess_col for excess_col in excess_columns]
                    ] = [rigd_irvea, rhv_irvea]
                    data_row[
                        ["iRVEA_Ranges" + excess_col for excess_col in excess_columns]
                    ] = [rigd_irvea_ranges, rhv_irvea_ranges]

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

                    # since composite front grows after each iteration this call should be done for each iteration
                    base = baseADM(cf, reference_vectors)

                    # generates the next reference point for the decision phase
                    ranges, reference_point = gp.generateRanges4decision(
                        base, max_assigned_vector[0], problem.ideal, problem.nadir
                    )

                    data_row["reference_point"] = [reference_point]
                    data_row["preferred_ranges"] = [ranges]

                    # run algorithms with the new reference point
                    pref_int_rvea[2].response = pd.DataFrame(
                        [reference_point],
                        columns=pref_int_rvea[2].content["dimensions_data"].columns,
                    )
                    pref_int_rvea_ranges[3].response = ranges

                    previous_RVEA_FEs = int_rvea._function_evaluation_count
                    previous_RVEA_Ranges_FEs = (
                        int_rvea_ranges._function_evaluation_count
                    )
                    _, pref_int_rvea = int_rvea.iterate(pref_int_rvea[2])
                    _, pref_int_rvea_ranges = int_rvea_ranges.iterate(
                        pref_int_rvea_ranges[3]
                    )

                    peritr_RVEA_FEs = (
                        int_rvea._function_evaluation_count - previous_RVEA_FEs
                    )
                    peritr_RVEA_Ranges_FEs = (
                        int_rvea_ranges._function_evaluation_count
                        - previous_RVEA_Ranges_FEs
                    )
                    # extend composite front with newly obtained solutions
                    (
                        rvea_n_solutions,
                        rvea_ranges_n_solutions,
                        cf,
                    ) = generate_composite_front_with_identity(
                        int_rvea.population.objectives,
                        int_rvea_ranges.population.objectives,
                        cf,
                    )

                    data_row["iRVEA_RP_N_Ss"] = [rvea_n_solutions]
                    data_row["iRVEA_Ranges_N_Ss"] = [rvea_ranges_n_solutions]
                    data_row["iRVEA_RP_FEs"] = [peritr_RVEA_FEs * n_obj]
                    data_row["iRVEA_Ranges_FEs"] = [peritr_RVEA_Ranges_FEs * n_obj]

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

                    rvea_ranges_transformer = Normalizer().fit(
                        int_rvea_ranges.population.objectives
                    )
                    norm_rvea_ranges = rvea_ranges_transformer.transform(
                        int_rvea_ranges.population.objectives
                    )

                    rigd_irvea, rhv_irvea = rmetric.calc(
                        norm_rvea, others=norm_rvea_ranges
                    )
                    rigd_irvea_ranges, rhv_irvea_ranges = rmetric.calc(
                        norm_rvea_ranges, others=norm_rvea
                    )

                    data_row[
                        ["iRVEA_RP" + excess_col for excess_col in excess_columns]
                    ] = [rigd_irvea, rhv_irvea]
                    data_row[
                        ["iRVEA_Ranges" + excess_col for excess_col in excess_columns]
                    ] = [rigd_irvea_ranges, rhv_irvea_ranges]

                    data = data.append(data_row, ignore_index=1)

    data.to_csv(
        f"./results/extendedADM/RVEA_RPvsRanges/100_generations/output{run+1}.csv",
        index=False,
    )

