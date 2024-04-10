#% ------------------------------------------------------------------------%
#% This is the main function to sample expected number of weight vectors on 
#% the PF of a given test problem.
#%
#% Author: Dr. Ke Li
#% Affliation: CODA Group @ University of Exeter
#% Contact: k.li@exeter.ac.uk || https://coda-group.github.io/
#% ------------------------------------------------------------------------%
import numpy as np
from desdeo_emo.utilities.initweight import initweight
from desdeo_emo.utilities.multi_layer_weight import multi_layer_weight
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymoo.problems.many.wfg import WFG1,WFG2,WFG3,WFG4,WFG5,WFG6,WFG7,WFG8,WFG9
from pymoo.util.ref_dirs import get_reference_directions

def samplingIGD(objDim:int, no_layers, no_gaps, shrink_factors, sample_size, id_problem):

    #%% generate the reference vectors
    if (objDim < 15):
        W = initweight(objDim, sample_size)
        W = W.T
    else:
        W = multi_layer_weight(objDim, no_layers, no_gaps, shrink_factors)
    
    #%% ZDT1
    if (id_problem == 1):
        P       = np.zeros((sample_size, objDim))
        f1      = np.linspace(0, 1, sample_size)
        P[:, 0] = f1
        P[: ,1] = np.ones(sample_size) - np.sqrt(P[:, 0])
    #%% ZDT2
    elif (id_problem == 2):
        P       = np.zeros((sample_size, objDim))
        f1      = np.linspace(0, 1, sample_size)
        P[:, 0] = f1
        P[: ,1] = np.ones(sample_size) - np.power(P[:, 0], 2)
    #%% ZDT3
    elif (id_problem == 3):
        f1      = np.linspace(0, 1, sample_size)
        P[:, 0] = f1
        P[:, 1] = 1 - np.sqrt(f1) - np.power(f1, np.sin(10 * np.pi * f1))
        P       = find_nondominated(P, 2)
    #%% DTLZ1
    elif (id_problem == 4):
        denominator = np.sum(W, axis = 1)
        #print("falla ridicula",np.ones((objDim, 1)))
        de_matrix = denominator[:, np.newaxis]  # Reshape to align for broadcasting
        P = W / (2 * de_matrix)
        #P = P.T
    #%% DTLZ2 - DTLZ4
    elif (id_problem == 5):
        tempW = W * W;
        denominator = np.sum(tempW, axis = 1)
        #print("falla ridicula",np.ones((objDim, 1)))
        de_matrix = np.tile(denominator[:, np.newaxis], (1, objDim))
        P = W / np.sqrt(de_matrix)
    elif (id_problem == 6):
        theta = np.arange(0, 1 + 1 / (sample_size - 1), 1 / (sample_size - 1))
        f1    = np.cos(theta * np.pi / 2) * np.cos(np.pi / 4)
        f2    = np.cos(theta * np.pi / 2) * np.sin(np.pi / 4)
        f3    = np.sin(theta * np.pi / 2)
        
        P = np.zeros((sample_size, objDim))
        P[:, 0] = f1
        P[:, 1] = f2
        P[:, 2] = f3

    #%% DTLZ7
    elif (id_problem == 7):
        step = int(np.sqrt(sample_size))
        step = int(np.sqrt(sample_size))
        f1 = np.arange(0, 1 + 1 / (step - 1), 1 / (step - 1))
        f2 = f1
        P = np.zeros((step * step, objDim))
        for i in range(step):
            for j in range(step):
                idx = i * step + j
                P[idx, 0] = f1[i]
                P[idx, 1] = f2[j]
        t1 = P[:, 0] * (np.ones(step * step) + np.sin(3 * np.pi * P[:, 0]))
        t2 = P[:, 1] * (np.ones(step * step) + np.sin(3 * np.pi * P[:, 1]))
        t3 = 3 - t1 - t2  # Calculate the third objective separately
        P = np.column_stack((P, t3)) 
        P = find_nondominated(P.T, objDim)
        # Plot the data points
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(P[:,0], P[:,1], P[:,2], color='blue', label='Data points')
        plt.show()
    elif (id_problem == 8):
        P       = np.zeros((sample_size, objDim))
        f1      = np.linspace(0, 1, sample_size)
        P[:, 0] = f1
        P[: ,1] = np.ones((sample_size, 1)) - P[:, 0]
    #WFG1
    elif (id_problem == 9):
        if (objDim == 3):
            wfg = WFG1(n_var=14, n_obj=3)
            ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)

        elif (objDim == 5):
            wfg = WFG1(n_var=18, n_obj=5)
            ref_dirs = get_reference_directions("uniform", 5, n_partitions=6)
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)

        elif (objDim ==7):
            wfg = WFG1(n_var=22, n_obj=7)
            ref_dirs = get_reference_directions(
                "multi-layer",
                get_reference_directions("das-dennis", 7, n_partitions=3, scaling=1.0),
                get_reference_directions("das-dennis", 7, n_partitions=2, scaling=0.5)
            )
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)
        else:
            print('Bad id!')
    #WFG2
    elif (id_problem == 10):
        if (objDim == 3):
            wfg = WFG2(n_var=14, n_obj=3)
            ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)

        elif (objDim == 5):
            wfg = WFG2(n_var=18, n_obj=5)
            ref_dirs = get_reference_directions("uniform", 5, n_partitions=6)
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)

        elif (objDim ==7):
            wfg = WFG2(n_var=22, n_obj=7)
            ref_dirs = get_reference_directions(
                "multi-layer",
                get_reference_directions("das-dennis", 7, n_partitions=3, scaling=1.0),
                get_reference_directions("das-dennis", 7, n_partitions=2, scaling=0.5)
            )
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)
        else:
            print('Bad id!')
    #WFG3
    elif (id_problem == 11):
        if (objDim == 3):
            wfg = WFG3(n_var=14, n_obj=3)
            ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)

        elif (objDim == 5):
            wfg = WFG3(n_var=18, n_obj=5)
            ref_dirs = get_reference_directions("uniform", 5, n_partitions=6)
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)

        elif (objDim ==7):
            wfg = WFG3(n_var=22, n_obj=7)
            ref_dirs = get_reference_directions(
                "multi-layer",
                get_reference_directions("das-dennis", 7, n_partitions=3, scaling=1.0),
                get_reference_directions("das-dennis", 7, n_partitions=2, scaling=0.5)
            )
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)
        else:
            print('Bad id!')
    #WFG4
    elif (id_problem == 12):
        if (objDim == 3):
            wfg = WFG4(n_var=14, n_obj=3)
            ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)

        elif (objDim == 5):
            wfg = WFG4(n_var=18, n_obj=5)
            ref_dirs = get_reference_directions("uniform", 5, n_partitions=6)
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)

        elif (objDim ==7):
            wfg = WFG4(n_var=22, n_obj=7)
            ref_dirs = get_reference_directions(
                "multi-layer",
                get_reference_directions("das-dennis", 7, n_partitions=3, scaling=1.0),
                get_reference_directions("das-dennis", 7, n_partitions=2, scaling=0.5)
            )
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)
        else:
            print('Bad id!')
    #WFG5
    elif (id_problem == 13):
        if (objDim == 3):
            wfg = WFG5(n_var=14, n_obj=3)
            ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)

        elif (objDim == 5):
            wfg = WFG5(n_var=18, n_obj=5)
            ref_dirs = get_reference_directions("uniform", 5, n_partitions=6)
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)

        elif (objDim ==7):
            wfg = WFG5(n_var=22, n_obj=7)
            ref_dirs = get_reference_directions(
                "multi-layer",
                get_reference_directions("das-dennis", 7, n_partitions=3, scaling=1.0),
                get_reference_directions("das-dennis", 7, n_partitions=2, scaling=0.5)
            )
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)
        else:
            print('Bad id!')
    #WFG6
    elif (id_problem == 14):
        if (objDim == 3):
            wfg = WFG6(n_var=14, n_obj=3)
            ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)

        elif (objDim == 5):
            wfg = WFG6(n_var=18, n_obj=5)
            ref_dirs = get_reference_directions("uniform", 5, n_partitions=6)
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)

        elif (objDim ==7):
            wfg = WFG6(n_var=22, n_obj=7)
            ref_dirs = get_reference_directions(
                "multi-layer",
                get_reference_directions("das-dennis", 7, n_partitions=3, scaling=1.0),
                get_reference_directions("das-dennis", 7, n_partitions=2, scaling=0.5)
            )
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)
        else:
            print('Bad id!')
    #WFG7
    elif (id_problem == 15):
        if (objDim == 3):
            wfg = WFG7(n_var=14, n_obj=3)
            ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)

        elif (objDim == 5):
            wfg = WFG7(n_var=18, n_obj=5)
            ref_dirs = get_reference_directions("uniform", 5, n_partitions=6)
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)

        elif (objDim ==7):
            wfg = WFG7(n_var=22, n_obj=7)
            ref_dirs = get_reference_directions(
                "multi-layer",
                get_reference_directions("das-dennis", 7, n_partitions=3, scaling=1.0),
                get_reference_directions("das-dennis", 7, n_partitions=2, scaling=0.5)
            )
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)
        else:
            print('Bad id!')
    #WFG8
    elif (id_problem == 16):
        if (objDim == 3):
            wfg = WFG8(n_var=14, n_obj=3)
            ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)

        elif (objDim == 5):
            wfg = WFG8(n_var=18, n_obj=5)
            ref_dirs = get_reference_directions("uniform", 5, n_partitions=6)
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)

        elif (objDim ==7):
            wfg = WFG8(n_var=22, n_obj=7)
            ref_dirs = get_reference_directions(
                "multi-layer",
                get_reference_directions("das-dennis", 7, n_partitions=3, scaling=1.0),
                get_reference_directions("das-dennis", 7, n_partitions=2, scaling=0.5)
            )
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)
        else:
            print('Bad id!')
    #WFG9
    elif (id_problem == 17):
        if (objDim == 3):
            wfg = WFG9(n_var=14, n_obj=3)
            ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)

        elif (objDim == 5):
            wfg = WFG9(n_var=18, n_obj=5)
            ref_dirs = get_reference_directions("uniform", 5, n_partitions=6)
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)

        elif (objDim ==7):
            wfg = WFG9(n_var=22, n_obj=7)
            ref_dirs = get_reference_directions(
                "multi-layer",
                get_reference_directions("das-dennis", 7, n_partitions=3, scaling=1.0),
                get_reference_directions("das-dennis", 7, n_partitions=2, scaling=0.5)
            )
            P = wfg.pareto_front(use_cache=False, ref_dirs=ref_dirs)
        else:
            print('Bad id!')

    else:
        print('Bad id!')
    return P

#%% Find out the dominance relationship between 'a' and 'b'
def dominated_relationship(a, b, m):
#% Input Parameters :  a->ind1; b->ind2; m-># of objectives??
#% Output Parameters: 1->a dominates b; 2->b dominates a; 3->a equals b;
#% 4->a and b are non-dominated to each other
    t = 0
    q = 0
    p = 0
    #e = 0.00001
    for i in range(0, m):
        if (a[i] <= b[i]):
            t = t + 1
        if (a[i] >= b[i]):
            q = q + 1
        if (a[i] == b[i]):
            p = p + 1
    if (t == m and p != m):
        x = 1
    elif (q == m and p != m):
        x = 2
    elif (p == m):
        x = 3
    else:
        x = 4
    return x


#%% Find out the non-dominated solutions in 'POP'
def find_nondominated(POP, m):
#% Input Parameter : POP->population; m-># of objectives
#% Output Parameter: NPOP->non-dominated solutions
    i = 0
    while (i < np.size(POP, 0)):
        flag = 0
        j = i + 1
        while (j < np.size(POP, 0)):
            x = dominated_relationship(POP[i, :], POP[j, :], m)
            if (x == 2):
                flag = 1
                break
            elif (x == 3):
                POP[j, :] = []
            elif (x == 1):
                POP[j, :] = []
            else:
                j = j + 1
        if (flag == 1):
            POP[i, :] = []
        else:
            i = i + 1
    NPOP = POP
    return NPOP
