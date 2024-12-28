#% ------------------------------------------------------------------------%
#% This is the function to sample weight vecotrs using multiple-layer method
#%
#% Author: Dr. Ke Li
#% Affliation: CODA Group @ University of Exeter
#% Contact: k.li@exeter.ac.uk || https://coda-group.github.io/
#% ------------------------------------------------------------------------%
from desdeo_emo.utilities.initweight import initweight
import numpy as np
from scipy.special import comb

def multi_layer_weight(objDim, no_layers, no_gaps, shrink_factors):
    layer_sizes = np.zeros(no_layers)
    
    #%% get the number of sample size on each layer
    for i in range (0, no_layers):
        layer_sizes[i] = comb(objDim + no_gaps[i] - 1, no_gaps[i])

    #%% weight vectors in the first layer
    W = initweight(objDim, layer_sizes[0])
    #W = cur_layer
    for i in range(1, no_layers):
        #%% generate a temporary layer
        temp_layer = initweight(objDim, layer_sizes[i])
        #%% shrink the temporary layer (coordinate transformation)
        cur_layer = (1 - shrink_factors[i]) / objDim * np.ones((objDim, layer_sizes[i])) + shrink_factors[i] * temp_layer
        #%% incorporate the current layer into the whole weight vector set
        W = np.append(W, cur_layer.T, 0)
    return W
