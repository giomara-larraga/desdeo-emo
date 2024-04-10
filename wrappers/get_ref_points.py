#! /usr/bin/env python3

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from desdeo_emo.utilities.samplingIGD import samplingIGD
from pymoo.problems.many.wfg import WFG1,WFG2,WFG3,WFG4,WFG5,WFG6,WFG7,WFG8,WFG9
from pymoo.util.ref_dirs import get_reference_directions


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
    


    num_points = PF.shape[0]

    for i in range(num_points):
        if all(PF[i, :] >= ref_point) and any(PF[i, :] > ref_point):
            return "Dominated"

    return "Non-dominated"


if __name__ == "__main__":
    #problem_name = "DTLZ1"
    #id_problem = 6   #DTLZ1 =4, DTLZ2-4 = 5, DTLZ5-6=6
    #objectives = 9
    #
    #no_layers = 2                  # number of layers
    #no_gaps   = [3, 2]             # specify the # of divisions on each layer
    #shrink_factors = [1.0, 0.5]    # shrinkage factor for each layer
    #sample_size = 10000

    #ref_point = [0.5,1.0,0.7]
    
    wfg = WFG9(n_var=22, n_obj=7)
    ref_dirs = get_reference_directions(
        "multi-layer",
        get_reference_directions("das-dennis", 7, n_partitions=3, scaling=1.0),
        get_reference_directions("das-dennis", 7, n_partitions=2, scaling=0.5)
    )
    PF = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)

    ideal = (np.max(PF, axis=0))
    nadir = (np.min(PF, axis=0))

    print(ideal)
    print(nadir)
    #ref_point =[0.0321854, 0.00114581, 0.02769762, 0.0538603, 0.52791254, 0.01432793, 0.07759848]
    ban=True
    #print((ideal-nadir)/2)
    while (ban):
        ref_point = np.random.uniform(nadir, ideal)
        print(ref_point)
        #PF = samplingIGD(objectives, no_layers, no_gaps, shrink_factors, sample_size, id_problem)

        result = is_dominated(ref_point, PF)
        if result == "Non-dominated":
            ban=False
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

    #WFG1 non-dominated
    #k=3 [1.23910631 2.44637419 5.5527944 ]
    #k=5 [1.03585518 0.43659334 2.14230848 2.16674522 2.29941606]
    #k=7 [ 1.76688957  3.09906177  1.65588859  1.0546838   9.38925089 11.03827762 6.82973547]
    #k=9

    #WFG1 dominated
    #k=3 [0.15375682 1.77951325 0.64890401]
    #k=5 [0.05125399 0.46175564 0.42769457 0.24440359 0.82026726]
    #k=7 [0.0321854, 0.00114581, 0.02769762, 0.0538603, 0.52791254, 0.31432793, 0.37759848]
    #k=9

    #WFG2 non-dominated
    #k=3 [1.30967692 0.99346475 4.47116006]
    #k=5 [0.35021854 3.01944369 4.20524078 7.21112237 6.21163321]
    #k=7 [0.929313   0.59260841 2.53331044 6.51579278 6.77173856 9.71325517 8.06674592]
    #k=9

    #WFG2 dominated
    #k=3 [0.14622299 0.34191758 0.74978728]
    #k=5 [0.13891394 0.01697542 0.40556752 0.10751129 2.0869258 ]
    #k=7 [0.0321854, 0.00114581, 0.02769762, 0.0538603, 0.52791254, 0.01432793, 0.07759848]
    #k=9

    #WFG3 non-dominated
    #k=3 [0.5630203  1.79517005 5.33063833]
    #k=5 [0.11885855 0.21359209 0.90970021 2.49710585 6.96706673]
    #k=7 [ 0.0184367   0.02499648  0.08673249  0.28976292  0.6339428   1.77542058 10.39965796]
    #k=9

    #WFG3 dominated
    #k=3 [0.30092965 0.24627122 1.42768285]
    #k=5 [0.0704111  0.16286383 0.49616984 0.20910561 4.64589562]
    #k=7 [0.0021854, 0.00114581, 0.00769762, 0.0038603, 0.00091254, 0.00432793, 0.00759848]
    #k=9

    #WFG4 non-dominated
    #k=3 [1.31993029 3.41135828 4.5660427 ]
    #k=5 [1.43438698 1.05028239 0.9828054  0.09652404 9.0034773 ]
    #k=7 [0.88561501 0.52641123 0.26153439 4.01412823 1.02227364 2.70903415 3.02380693]
    #k=9

    #WFG4 dominated
    #k=3 [1.06663281 1.55156518 1.78413639]
    #k=5 [0.72758636 2.27700352 2.14589851 0.9682322  2.72728607]
    #k=7 [0.0321854, 0.00114581, 0.02769762, 0.0538603, 0.52791254, 0.01432793, 0.07759848]
    #k=9

    #WFG5 non-dominated
    #k=3 [0.33235522 2.60061373 5.14478791]
    #k=5 [0.47606867 0.92775935 3.09163882 0.70410871 6.18430942]
    #k=7 [1.37011353 1.12749637 3.79669087 1.58726585 3.95467123 6.35961829 9.10177554]
    #k=9

    #WFG5 dominated
    #k=3 [0.33864139 1.75842047 4.07356541]
    #k=5 [1.18199819 1.33159513 2.01704019 1.04923519 0.88043244]
    #k=7 [0.0321854, 0.00114581, 0.02769762, 0.0538603, 0.52791254, 0.01432793, 0.07759848]
    #k=9

    #WFG6 non-dominated
    #k=3 [1.07531107 2.46361268 4.99810657]
    #k=5 [1.05721529 1.16083727 0.55058563 5.10521206 4.03077472]
    #k=7 [1.97017758 0.62211934 3.03953557 2.69398312 0.57442746 1.6731152 5.80030565]
    #k=9

    #WFG6 dominated
    #k=3 [0.92761334 1.144132   3.43439072]
    #k=5 [0.02557613 1.02124019 3.20016649 1.73815541 5.36407484]
    #k=7 [0.0321854, 0.00114581, 0.02769762, 0.0538603, 0.52791254, 0.01432793, 0.07759848]
    #k=9

    #WFG7 non-dominated
    #k=3 [0.97091881 3.58944958 5.54646309]
    #k=5 [1.36038027 2.24846641 5.26738004 2.33953644 3.63775975]
    #k=7 [ 0.43169456  0.42602308  3.94251817  5.44343455  3.59233515 10.0311028 6.941378  ]
    #k=9

    #WFG7 dominated
    #k=3 [0.5576288  1.62558508 4.81924167]
    #k=5 [0.61231767 0.66957053 1.59509457 1.67185483 0.30701825]
    #k=7 [0.0321854, 0.00114581, 0.02769762, 0.0538603, 0.52791254, 0.01432793, 0.07759848]
    #k=9

    #WFG8 non-dominated
    #k=3 [1.59112024 3.25464029 4.22527344]
    #k=5 [0.83918722 1.84890951 5.57725313 5.20116428 2.643046  ]
    #k=7 [0.58080649 0.47314825 4.9939482  1.9847054  1.41471698 1.22996304 8.16889554]
    #k=9

    #WFG8 dominated
    #k=3 [0.59253235 0.61381295 2.6696158 ]
    #k=5 [0.47210955 1.26425429 1.80968167 4.19553464 3.45470379]
    #k=7 [0.0321854, 0.00114581, 0.02769762, 0.0538603, 0.52791254, 0.01432793, 0.07759848]
    #k=9

    #WFG9 non-dominated
    #k=3 [1.91920752 2.77433314 4.1798446 ]
    #k=5 [0.5787645  0.35715373 0.57047549 5.35221865 9.76036895]
    #k=7 [ 0.91792565  2.68347427  4.69633084  7.58069192  3.76690163 11.12729828 2.24569395]
    #k=9

    #WFG9 dominated
    #k=3 [0.35289168 2.86170828 2.58751479]
    #k=5 [1.40552173 0.89685861 0.03326243 0.28571957 3.09879354]
    #k=7 [0.0321854, 0.00114581, 0.02769762, 0.0538603, 0.52791254, 0.01432793, 0.07759848]
    #k=9
    