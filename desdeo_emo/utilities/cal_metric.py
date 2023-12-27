#% ------------------------------------------------------------------------%
#% This function is used to calculate the IGD and HV value of the solution
#% set processed by R-metric.
#%
#% Author: Dr. Ke Li
#% Affliation: CODA Group @ University of Exeter
#% Contact: k.li@exeter.ac.uk || https://coda-group.github.io/
#%
#% Reference: K. Li, K. Deb and X. Yao, "R-Metric: Evaluating the 
#% Performance of Preference-Based Evolutionary Multi-Objective Optimization 
#% Using Reference Points", IEEE Trans. on Evol. Comput., accepted for
#% publication, July 2017.
#% ------------------------------------------------------------------------%
import numpy as np
from desdeo_emo.utilities.hv import HyperVolume

def cal_metric(pop, PF, w_point, popsize, PFsize):
    if (popsize == 0): #% if there is no useful solution, IGD and HV is -1
        IGD = -1
        HV  = -1
    else:
        #%% IGD computation
        min_dist = np.zeros(PFsize)
        for j in range(0, PFsize):
            temp    = PF[j, :]
            tempMat = np.repeat([temp], popsize, axis = 0)
            temp_dist   = (tempMat - pop) ** 2
            distance    = np.sum(temp_dist, axis = 1)
            min_dist[j] = np.min(distance)
        min_dist = np.sqrt(min_dist)
        IGD      = np.mean(min_dist)

        #%% HV computation
        hv = HyperVolume(w_point)
        HV = hv.compute(pop)
    return [IGD, HV]
