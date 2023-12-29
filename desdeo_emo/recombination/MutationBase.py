from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd


class MutationBase(ABC):
    """The base class for the selection operator."""

    @abstractmethod
    def do(self, *args) -> List[int]:
        """Use the selection operator over the given fitness values. Return the indices
            individuals with the best fitness values according to the operator.

        Parameters
        ----------
        fitness : np.ndarray
            Fitness of the individuals from which the next generation is to be selected.

        Returns
        -------
        List[int]
            The list of selected individuals
        """
    def repair(self, value, method, lower_bounds, upper_bounds):
        #lower_bounds = self.pop.problem.get_variable_lower_bounds()
        #upper_bounds = self.pop.problem.get_variable_upper_bounds()
        value_repaired = value.copy()
        if (method =="random"):
            for i in range(len(value)):
                if (value[i] < lower_bounds[i]) or (value[i] > upper_bounds[i]):
                    range_size = (upper_bounds[i] - lower_bounds[i])  # 2
                    value_repaired[i] = np.random.rand() * range_size + lower_bounds[i]            
        elif (method =="round"):
            for i in range(len(value)):
                if (value[i] < lower_bounds[i]):
                    value_repaired[i] = upper_bounds[i]
                if (value[i] > upper_bounds[i]):
                    value_repaired[i] = lower_bounds[i]
        elif (method == "bounds"):
            for i in range(len(value)):
                if (value[i] < lower_bounds[i]):
                    value_repaired[i] = lower_bounds[i]
                if (value[i] > upper_bounds[i]):
                    value_repaired[i] = upper_bounds[i]
        
        return value_repaired
