import argparse
import logging
import sys

import numpy as np
import pandas as pd
from desdeo_emo.utilities.filter_NDS import filter_NDS
from scipy.stats import ranksums
from desdeo_emo.utilities.pf_samples import pf_samples
from desdeo_emo.utilities.preprocessing_asf import preprocessing_asf
from desdeo_emo.utilities.cal_metric import cal_metric
from desdeo_problem import MOProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.EAs.RNSGAII import RNSGAII
from desdeo_emo.EAs.RNSGAIII import RNSGAIII

def main(ALG, PROBNAME, POP, CXPB, MUTPB, DATFILE):
    problem = test_problem_builder(PROBNAME)
    # only useful for the many-objective scenario (i.e., objDim > 3)
    no_layers = 2                  # number of layers
    no_gaps   = [3, 2]             # specify the # of divisions on each layer
    shrink_factors = [1.0, 0.5]    # shrinkage factor for each layer
    igdsamSize = 10000
    # set trimming radius
    radius = 0.2
    if (ALG == "RNSGA2"):
        evolver = RNSGAII(
            problem,
            n_iterations=1,
            n_gen_per_iter=100,
            population_size=100,
            interact=True,
            epsilon=0.001,
            normalization="front",
            weights=None,
            extreme_points_as_reference_points=False,
        )
        evolver.set_interaction_type("Reference point")
        responses = np.asarray([[0.5, 0.5]])
        pref, plot = evolver.start()
        pref.response = pd.DataFrame(
            [responses[0]], columns=pref.content["dimensions_data"].columns
        )
        pref, plot = evolver.iterate(pref)
        i2, obj = evolver.end()
    elif (ALG == "RNSGA3"):
        evolver = RNSGAIII(
            problem,
            n_iterations=1,
            n_gen_per_iter=100,
            population_size=100,
            interact=True,
            mu=0.5,
            save_non_dominated=True,
        )
        evolver.set_interaction_type("Reference point")
        responses = np.asarray([[0.5, 0.5]])
        pref, plot = evolver.start()
        pref.response = pd.DataFrame(
            [responses[0]], columns=pref.content["dimensions_data"].columns
        )
        pref, plot = evolver.iterate(pref)
        i2, obj, base = evolver.end()
    else:
        print(ALG)


    if len(obj)>0:
        w_point = responses[0] + 2 * np.ones(obj.shape[1])
        PF, PFsize = pf_samples(obj.shape[1], no_layers, no_gaps, shrink_factors, igdsamSize, 1, radius, responses[0], w_point)
        RNSGA2, RNSGA2_size    = preprocessing_asf(obj, responses[0], w_point, radius)
        RNSGA2_IGD, RNSGA2_HV   = cal_metric(RNSGA2, PF, w_point, RNSGA2_size, PFsize)
        print(RNSGA2_IGD)
    else:
        RNSGA2_IGD = 0


    # save the fo values in DATFILE
    with open(DATFILE, 'w') as f:
    	f.write(str(RNSGA2_IGD))

if __name__ == "__main__":
    # just check if args are ok
    with open('args.txt', 'w') as f:
        f.write(str(sys.argv))
    
    # loading example arguments
    ap = argparse.ArgumentParser(description='Feature Selection using GA with DecisionTreeClassifier')
    ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    # 3 args to test values
    ap.add_argument('--alg', dest='alg', type=str, required=True, help='Algorithm name')
    ap.add_argument('--prob', dest='prob', type=str, required=True, help='Problem name')
    ap.add_argument('--pop', dest='pop', type=int, required=False, help='Population size')
    ap.add_argument('--cros', dest='cros', type=float, required=False, help='Crossover probability')
    ap.add_argument('--mut', dest='mut', type=float, required=False, help='Mutation probability')
    # 1 arg file name to save and load fo value
    ap.add_argument('--datfile', dest='datfile', type=str, required=False, help='File where it will be save the score (result)')

    args = ap.parse_args()
    logging.debug(args)
    # call main function passing args
    main(args.alg, args.prob, args.pop, args.cros, args.mut, args.datfile)