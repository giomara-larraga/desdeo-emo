#% ------------------------------------------------------------------------%
#% This function preprocesses the population before R-metric computation,
#% i.e., 1. Filters the irrelevant solutions; 2. Translate the trimmed
#% solutions to the virture position.
#%
#% Author:  Dr. Ke Li @ University of Exeter
#% Contact: k.li@exeter.ac.uk (https://coda-group.github.io/)
#% Last modified: 25/12/2016
#% ------------------------------------------------------------------------%
import numpy as np
from desdeo_emo.utilities.trim_cubic import trim_cubic

def preprocessing_asf(data, ref_point, w_point, radius):

    datasize = np.size(data, 0)

    if (datasize == 0):
        new_data = data
        new_size = datasize
        return [new_data, new_size]

    #%% Step 1: identify representative point
    ref_matrix  = np.repeat([ref_point], datasize, axis = 0)
    w_matrix    = np.repeat([w_point], datasize, axis = 0)
    diff_matrix = np.divide((data - ref_matrix), (w_matrix - ref_matrix))
    agg_value   = np.max(diff_matrix, axis = 1)
    idx         = np.argmin(agg_value)
    zp          = data[idx, :]

    #%% Step 2: trim data
    trimed_data = trim_cubic(data, zp, radius) #% trim as a cubic
    trimed_size = np.size(trimed_data, 0)

    #%% Step 3: transfer trimmed data to the reference line
    #% find k
    temp = np.divide((zp - ref_point), (w_point - ref_point))
    kIdx = np.argmax(temp)

    #% find zl
    temp = (zp[kIdx] - ref_point[kIdx]) / (w_point[kIdx] - ref_point[kIdx])
    zl   = ref_point + temp * (w_point - ref_point)

    #% solution transfer
    temp = zl - zp
    shift_direction = np.repeat([temp], trimed_size, axis = 0)
    new_data  = trimed_data + shift_direction
    new_size  = np.size(new_data, 0)
    return [new_data, new_size]
