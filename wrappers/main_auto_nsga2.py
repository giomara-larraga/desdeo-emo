#! /usr/bin/env python3

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from desdeo_emo.utilities.filter_NDS import filter_NDS
from scipy.stats import ranksums
from desdeo_emo.utilities.samplingIGD import samplingIGD
from desdeo_emo.utilities.preprocessing_asf import preprocessing_asf
from desdeo_emo.utilities.cal_metric import cal_metric
from desdeo_problem import MOProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.EAs.AutoNSGAII import AutoNSGAII


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main(SEED, PROB, ID, OBJ, VAR, POP, GENS, CROS, CROS_PROB, CROS_REP, CROS_DIST, CROS_ALPHA, MUT, MUT_PROB, MUT_REPAIR, MUT_PMD, MUT_UMP, SEL, SEL_SIZE):
    problem = test_problem_builder(name=PROB, n_of_variables=VAR, n_of_objectives= OBJ)
    no_layers = 2                  # number of layers
    no_gaps   = [3, 2]             # specify the # of divisions on each layer
    shrink_factors = [1.0, 0.5]    # shrinkage factor for each layer
    # only useful for the many-objective scenario (i.e., objDim > 3)
    igdsamSize = 10000
        
    evolver = AutoNSGAII(
        problem,
        n_iterations=1,
        n_gen_per_iter=GENS,
        population_size=POP,
        seed= SEED,
        selection_parents = SEL,
        slection_tournament_size = SEL_SIZE,
        crossover = CROS,
        crossover_probability = CROS_PROB,
        crossover_distribution_index = CROS_DIST,
        crossover_repair = CROS_REP,
        blx_alpha_crossover = CROS_ALPHA,
        mutation = MUT,
        mutation_probability = MUT_PROB,
        mutation_repair = MUT_REPAIR,
        uniform_mut_perturbation  = MUT_UMP,
        polinomial_mut_dist_index = MUT_PMD,
    )
    
    evolver.start()

    while evolver.continue_evolution():
        evolver.iterate()
        
    i2, obj_values = evolver.end()

    if len(obj_values)>0:
        PF = samplingIGD(OBJ, no_layers, no_gaps, shrink_factors, igdsamSize, ID)
        PFsize   = np.shape(PF)[0]
        AutoNSGA2_Size = np.shape(obj_values)[0]

        AutoNSGA2_IGD   = cal_metric(obj_values, PF, None, AutoNSGA2_Size, PFsize)
        print(AutoNSGA2_IGD)

    else:
        AutoNSGA2_IGD = 0

    #fig = plt.figure(figsize=(8, 6))
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(obj_values[:,0], obj_values[:,1], obj_values[:,2], color='blue', label='Data points')
    #plt.show()

if __name__ == "__main__":
    # just check if args are ok
    with open('args.txt', 'w') as f:
        f.write(str(sys.argv))
    
    # loading example arguments
    ap = argparse.ArgumentParser(description='Feature Selection using GA with DecisionTreeClassifier')
    ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    # 3 args to test values
    ap.add_argument('--seed', dest='seed', type=int, required=True, help='Seed for random numbers')
    ap.add_argument('--prob', dest='prob', type=str, required=True, help='Problem name')
    ap.add_argument('--id', dest='id', type=int, required=True, help='Problem id (for r-metric)')
    ap.add_argument('--obj', dest='obj', type=int, required=True, help='Number of objectives')
    ap.add_argument('--var', dest='var', type=int, required=True, help='Number of variables')
    ap.add_argument('--populationSize', dest='pop', type=int, required=False, help='Population size')
    ap.add_argument('--generations', dest='gens', type=int, required=False, help='Number of generations')

    ap.add_argument('--crossover', dest='cros', type=str, required=False, help='Crossover type (SBX or BLX)')
    ap.add_argument('--crossoverProbability', dest='cros_prob', type=float, required=False, help='Crossover probability')
    ap.add_argument('--crossoverRepairStrategy', dest='cros_rep', type=str, required=False, help='Crossover repair strategy (RANDOM, ROUND, BOUNDS)')
    ap.add_argument('--sbxCrossoverDistributionIndex', dest='cros_dist', type=float, required=False, help='SBX Crossover distribution index')
    ap.add_argument('--blxAlphaCrossoverAlphaValue', dest='cros_alpha', type=float, required=False, help='BLX Crossover alpha value')

    ap.add_argument('--mutation', dest='mut', type=str, required=False, help='Mutation type ("polynomial, uniform")')
    ap.add_argument('--mutationProbability', dest='mut_prob', type=float, required=False, help='Mutation probability')
    ap.add_argument('--mutationRepairStrategy', dest='mut_repair', type=str, required=False, help='Mutation repair strategy (random, rpund, bounds)')
    ap.add_argument('--polynomialMutationDistributionIndex', dest='mut_pmd', type=float, required=False, help='Polynomial Mutation Distribution Index')
    ap.add_argument('--uniformMutationPerturbation', dest='mut_ump', type=float, required=False, help='Uniform Mutation Perturbation')

    ap.add_argument('--selection', dest='sel', type=str, required=False, help='Selection operator (random, tournament)')
    ap.add_argument('--selectionTournamentSize', dest='sel_size', type=int, required=False, help='Size of tournament selection')


    # 1 arg file name to save and load fo value
    #ap.add_argument('--datfile', dest='datfile', type=str, required=False, help='File where it will be save the score (result)')

    args = ap.parse_args()
    logging.debug(args)
    #np.random.seed(args.seed)
    # call main function passing args
    main(args.seed, args.prob, args.id, args.obj, args.var, args.pop, args.gens, args.cros,args.cros_prob, args.cros_rep, args.cros_dist, args.cros_alpha, args.mut, args.mut_prob, args.mut_repair, args.mut_pmd, args.mut_ump, args.sel, args.sel_size)
