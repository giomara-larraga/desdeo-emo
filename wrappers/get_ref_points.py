#! /usr/bin/env python3

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from desdeo_emo.utilities.samplingIGD import samplingIGD

def is_dominated(ref_point, PF):
    """
    Checks if ref_point is dominated or non-dominated with respect to PF.

    Parameters:
    ref_point (list or numpy array): The reference point to be evaluated.
    PF (numpy array): The Pareto front.

    Returns:
    str: "Dominated" if ref_point is dominated, "Non-dominated" otherwise.
    """

    ref_point = np.array(ref_point)
    PF = np.array(PF)

    num_points = PF.shape[0]

    for i in range(num_points):
        if all(PF[i, :] <= ref_point) and any(PF[i, :] < ref_point):
            return "Dominated"

    return "Non-dominated"


if __name__ == "__main__":
    problem_name = "DTLZ1"
    id_problem = 6   #DTLZ1 =4, DTLZ2-4 = 5, DTLZ5-6=6
    objectives = 9

    no_layers = 2                  # number of layers
    no_gaps   = [3, 2]             # specify the # of divisions on each layer
    shrink_factors = [1.0, 0.5]    # shrinkage factor for each layer
    sample_size = 10000

    ref_point = [0.6,0.7,0.8,0.4,0.8,0.8,0.7,0.5,0.6]
    PF = samplingIGD(objectives, no_layers, no_gaps, shrink_factors, sample_size, id_problem)

    result = is_dominated(ref_point, PF)
    print(f"The reference point is: {result}")

    #DTLZ1 non dominated
    #k=3  0.05,0.05,0.2
    #k=5  0.05,0.05,0.05,0.05,0.2
    #k=7  0.05,0.05,0.05,0.05,0.05,0.05,0.2
    #k=9  0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.2

    #DTLZ1 dominated
    #k=3 0.3,0.3,0.2
    #k=5 0.3,0.3,0.3, 0.3, 0.2
    #k=7 0.3,0.3,0.3,0.3,0.3,0.3,0.2
    #k=9 0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.2

    #DTLZ2-4 non dominated
    #k=3  0.2,0.5,0.6
    #k=5  0.2,0.5,0.5,0.2,0.6
    #k=7  0.2,0.5,0.5,0.2,0.2,0.5,0.6
    #k=9  0.2,0.5,0.2,0.2,0.2,0.2,0.2,0.5,0.6

    #DTLZ2-4 dominated
    #k=3 0.7,0.8,0.5
    #k=5 0.7,0.7,0.8,0.9,0.5
    #k=7 0.7,0.5,0.7,0.8,0.9,0.6,0.5
    #k=9 0.7,0.4,0.5,0.7,0.8,0.5,0.9,0.6,0.5

    #DTLZ5-6 non dominated
    #k=3  0.1,0.3,0.5
    #k=5  0.1,0.3,0.2,0.4,0.5
    #k=7  0.1,0.3,0.1,0.1,0.2,0.4,0.5
    #k=9  0.1,0.3,0.1,0.3,0.5,0.1,0.2,0.4,0.5

    #DTLZ5-6 dominated
    #k=3 0.6,0.7,0.6
    #k=5 0.6,0.7,0.8,0.4,0.6
    #k=7 0.6,0.7,0.8,0.4,0.8,0.8,0.6
    #k=9 0.6,0.7,0.8,0.4,0.8,0.8,0.7,0.5,0.6