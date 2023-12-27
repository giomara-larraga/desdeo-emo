#% ------------------------------------------------------------------------%
#% This function is used to obtain the PF samples for calculating the R-IGD.
#% Basically, it at first samples a set of PF points from the whole PF.
#% Then, it trims the data according to the DM's preference information.
#%
#% Author: Dr. Ke Li
#% Affliation: CODA Group @ University of Exeter
#% Contact: k.li@exeter.ac.uk || https://coda-group.github.io/
#% ------------------------------------------------------------------------%
import numpy as np
from desdeo_emo.utilities.samplingIGD import samplingIGD
from desdeo_emo.utilities.trim_cubic import trim_cubic

def pf_samples(objDim, no_layers, no_gaps, shrink_factors, igdsamSize, problem_id, radius, ref_point, w_point):
    #% sample a set of points from the whole PF
    IGD_reference = samplingIGD(objDim, no_layers, no_gaps, shrink_factors, igdsamSize, problem_id)
    igdsamSize    = np.shape(IGD_reference)[0]
    
    #% find the representative point in the set
    ref_matrix  = np.repeat([ref_point], igdsamSize, axis = 0)
    w_matrix    = np.repeat([w_point], igdsamSize, axis = 0)
    diff_matrix = np.divide((IGD_reference - ref_matrix), (w_matrix - ref_matrix))
    agg_value   = np.max(diff_matrix, axis=1)
    idx         = np.argmin(agg_value)
    target      = IGD_reference[idx, :]
    
    #% find the points used to calculate the R-IGD
    PF     = trim_cubic(IGD_reference, target, radius) #% trim as a cubic
    PFsize = np.shape(PF)[0]
    return PF, PFsize
