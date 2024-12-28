#% ------------------------------------------------------------------------%
#% This script is used to calculated the R-metric, R-IGD and R-HV in
#% particular, for preference-based EMO algorithms. Note that herer we only
#% consider the user preference elicited as a aspiration level vector in the
#% objective space.
#% 
#% Author: Dr. Ke Li
#% Affliation: CODA Group @ University of Exeter
#% Contact: k.li@exeter.ac.uk || https://coda-group.github.io/
#%
#% Reference: K. Li, K. Deb and X. Yao, "R-Metric: Evaluating the 
#% Performance of Preference-Based Evolutionary Multi-Objective Optimization 
#% Using Reference Points", IEEE Trans. on Evol. Comput., accepted for
#% publication, July 2017.
#
# Python version: Giomara L\'arraga
#% ------------------------------------------------------------------------%

import numpy as np
from desdeo_emo.utilities.filter_NDS import filter_NDS
from scipy.stats import ranksums
from desdeo_emo.utilities.pf_samples import pf_samples
from desdeo_emo.utilities.preprocessing_asf import preprocessing_asf
from desdeo_emo.utilities.cal_metric import cal_metric
# parameter settings

problems   = 'ZDT1'
refp_index = 0
objDim     = 2

problem_id = 1 # id = 1: ZDT1,4; id = 2: ZDT2,6; id = 3: ZDT3; id = 4: DTLZ1; id = 5: DTLZ2-4; id =6: DTLZ5,6; id = 7: DTLZ7
numRun     = 31
igdsamSize = 10000

# only useful for the many-objective scenario (i.e., objDim > 3)
no_layers = 2                  # number of layers
no_gaps   = [3, 2]             # specify the # of divisions on each layer
shrink_factors = [1.0, 0.5]    # shrinkage factor for each layer

# load path
path1 = '../test_data/r-stm/' + problems + 'M' + str(objDim) + '_' + str(refp_index + 1) + '/' 
path2 = '../test_data/RNSGA2/' + problems +'M' + str(objDim) +'_' + str(refp_index + 1) + '/'

# aspiration level vector settings
if (problems == 'ZDT1'):
    refp_set = np.array([[0.3, 0.4], [0.65, 0.3]])
elif (problems == 'ZDT2'):
    refp_set = np.array([[0.2, 0.8], [0.9, 0.4]])
elif (problems, 'ZDT3'):
    refp_set = np.array([[0.15, 0.4], [0.4, 0.0]])
elif (problems == 'ZDT4'):
    refp_set = np.array([[0.3, 0.4], [0.65, 0.3]])
elif (problems == 'ZDT6'):
    refp_set = np.array([[0.9, 0.3], [ 0.5, 0.7]])
elif (problems == 'DTLZ1'):
    refp_set = np.array([[0.05, 0.05, 0.2], [0.3, 0.3, 0.2]])
elif (problems == 'DTLZ2'):
    refp_set = np.array([[0.2, 0.5, 0.6], [0.7, 0.8, 0.5]])
elif (problems == 'DTLZ3'):
    refp_set = np.array([[0.2, 0.5, 0.6], [0.7, 0.8, 0.5]])
elif (problems == 'DTLZ4'):
    refp_set = np.array([[0.2, 0.5, 0.6], [0.7, 0.8, 0.5]])
elif (problems == 'DTLZ5'):
    refp_set = np.array([[0.1, 0.3, 0.5], [0.6, 0.7, 0.5]])
elif (problems == 'DTLZ6'):
    refp_set = np.array([[0.1, 0.3, 0.5], [0.6, 0.7, 0.5]])
elif (problems == 'DTLZ7'):
    refp_set = np.array([[0.165, 0.71, 4.678], [0.75, 0.15, 6.0]])

ref_point = refp_set[refp_index, :]

# set worst point
w_point = ref_point + 2 * np.ones(objDim)

# set trimming radius
radius = 0.2

# sample PF points given test problem
PF, PFsize = pf_samples(objDim, no_layers, no_gaps, shrink_factors, igdsamSize, problem_id, radius, ref_point, w_point)

# initialize data structure (here we use R-NSGA-II, MOEA/D-STM as example)
RNSGA2_IGD = np.zeros(numRun)
RNSGA2_HV  = np.zeros(numRun)

STM_IGD = np.zeros(numRun)
STM_HV =np.zeros(numRun)

for i in range (0, numRun):
    # load data
    STM     = np.loadtxt(path1 + 'STM_FUN' + str(i))
    RNSGA2  = np.loadtxt(path2 + 'RNSGA2_FUN' + str(i))
    
    data = np.append(STM, RNSGA2, 0)

    # filter non-dominated data
    STM     = filter_NDS(STM, data)
    RNSGA2  = filter_NDS(RNSGA2, data)
        
    # preprocess filtered data
    STM, STM_size          = preprocessing_asf(STM, ref_point, w_point, radius);
    RNSGA2, RNSGA2_size    = preprocessing_asf(RNSGA2, ref_point, w_point, radius);
    
    # calculate R-IGD and R-HV
    STM_IGD[i], STM_HV[i]         = cal_metric(STM, PF, w_point, STM_size, PFsize);
    RNSGA2_IGD[i], RNSGA2_HV[i]   = cal_metric(RNSGA2, PF, w_point, RNSGA2_size, PFsize);

# extract effect R-metric values
STM_IGD     = STM_IGD[STM_IGD != -1]
RNSGA2_IGD  = RNSGA2_IGD[RNSGA2_IGD != -1]

STM_HV     = STM_HV[STM_HV != -1]
RNSGA2_HV  = RNSGA2_HV[RNSGA2_HV != -1]

# mean and std of R-IGD and R-HV
mean_IGD_STM = np.mean(STM_IGD)
std_IGD_STM  = np.std(STM_IGD)
mean_HV_STM  = np.mean(STM_HV)
std_HV_STM   = np.std(STM_HV)

mean_IGD_RNSGA2 = np.mean(RNSGA2_IGD)
std_IGD_RNSGA2  = np.std(RNSGA2_IGD)
mean_HV_RNSGA2  = np.mean(RNSGA2_HV)
std_HV_RNSGA2   = np.std(RNSGA2_HV)



# print result
print('MOEA/D-STM: R-IGD = ' + str(mean_IGD_STM) + '(', str(std_IGD_STM), ')' + ', R-HV = ' + str(mean_HV_STM) + '(' + str(std_HV_STM) + ')')
print('R-NSGA2: R-IGD = ' + str(mean_IGD_RNSGA2) + '(', str(std_IGD_RNSGA2) + ')' + ', R-HV = ' + str(mean_HV_RNSGA2) + '(' + str(std_HV_RNSGA2) + ')')
print('===================================================================')

# Wilcoxon rank sum test
#higd_array    = -1 * np.ones((2, 1))
#IGD_array     = np.append(mean_IGD_STM, mean_IGD_RNSGA2, 0)
#best_idx      = np.argmin(IGD_array)
#if (best_idx == 0):
#    higd_array[1] = ranksums(STM_IGD, RNSGA2_IGD)
#else:
#    higd_array[0] = ranksums(RNSGA2_IGD, STM_IGD)
#
#print('Wilcoxon rank sum test IGD: ' + str(higd_array[0]) + ', ' + str(higd_array[1]))
#
#hhv_array     = -1 * np.ones((2, 1))
#HV_array      = np.append(mean_HV_STM, mean_HV_RNSGA2, 0)
#best_idx      = np.argmax(HV_array)
#if (best_idx == 0):
#    hhv_array[1] = ranksums(STM_HV, RNSGA2_HV)
#else:
#    hhv_array[0] = ranksums(RNSGA2_HV, STM_HV)
#    
#print('Wilcoxon rank sum test HV : ' + str(hhv_array(1)) + ', ' + str(hhv_array(2)))
