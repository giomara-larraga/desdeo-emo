#% ------------------------------------------------------------------------%
#% This function trims the irrelevant solutions for R-metric computation.
#%
#% Author: Dr. Ke Li
#% Affliation: CODA Group @ University of Exeter
#% Contact: k.li@exeter.ac.uk || https://coda-group.github.io/
#% ------------------------------------------------------------------------%
import numpy as np

def trim_cubic(pop, centroid, values):
    popsize, objDim = np.shape(pop)
    centroid_matrix = np.repeat([centroid], popsize, axis = 0)
    
    diff_matrix = pop - centroid_matrix

    radius      = values / 2.0
    flag_matrix = abs(diff_matrix) < radius
    flag_sum    = np.sum(flag_matrix, 1)
    
    filtered_idx = flag_sum == objDim
    filtered_pop = pop[filtered_idx, :]
    return filtered_pop