#% ------------------------------------------------------------------------%
#% This function is used to filter the non-dominated solutions from the 
#% current considered population
#%
#% Author:  Dr. Ke Li @ University of Birmingham
#% Contact: keli.genius@gmail.com (http://www.cs.bham.ac.uk/~likw)
#% Last modified: 01/10/2016
#% ------------------------------------------------------------------------%
import numpy as np

#% This function is used to check the dominance relationship between a and b
#% 1: a dominates b | -1: b dominates a | 0: non-dominated
def check_dominance(a, b, nobj):
    
    flag1 = 0
    flag2 = 0
    
    for i in range (0, nobj):
        if (a[i] < b[i]):
            flag1 = 1
        else:
            if (a[i] > b[i]):
                flag2 = 1
    
    if ((flag1 == 1) and (flag2 == 0)):
        dominance_flag = 1
    elif ((flag1 == 0) and (flag2 == 1)):
        dominance_flag = -1
    else:
        dominance_flag = 0
    return dominance_flag


def filter_NDS(cur_pop, whole_pop):
    num_objs = np.shape(cur_pop)[1]
    index_array = np.zeros(np.shape(cur_pop)[0])
    for i in range (0, np.shape(cur_pop)[0]):
        for j in range (0, np.shape(whole_pop)[0]):
            flag = check_dominance(cur_pop[i, :], whole_pop[j, :], num_objs)
            if (flag == -1):
                index_array[i] = 1
                break
    final_index = (index_array == 0)
    filtered_pop = cur_pop[final_index, :]
    return filtered_pop

