import numpy as np
from desdeo_emo.recombination.MutationBase import MutationBase
import math

class BP_mutation(MutationBase):
    def __init__(
        self,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        ProM: float = None,
        DisM: float = 20,
        repair_method: str = "bound",
    ):
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        if ProM is None:
            self.ProM = 1/len(lower_limits)
        else:
            self.ProM = ProM
        self.DisM = DisM
        self.repair_method = repair_method

    def do(self, offspring: np.ndarray):
        #print("toda cresta")
        result = offspring.copy()
        for i in range(len(offspring)):
            for j in range(len(offspring[i])):
                y = offspring[i][j]
                if np.random.rand() <= self.ProM:
                    yl = self.lower_limits[j]
                    yu = self.upper_limits[j]
                
                    if yl == yu:
                        y = yl
                    else:
                        delta1 = (y - yl) / (yu - yl)
                        delta2 = (yu - y) / (yu - yl)
                        rnd = np.random.rand()
                        mut_pow = 1.0 / (self.DisM + 1.0)
                        deltaq = 0.0
                        val = 0.0
                        xy = 0.0

                        if rnd <= 0.5:
                            xy = 1.0 - delta1
                            val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (math.pow(xy, self.DisM + 1.0))
                            print(val)
                            deltaq = math.pow(val, mut_pow) - 1.0
                        else:
                            xy = 1.0 - delta2
                            val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (math.pow(xy, self.DisM + 1.0))
                            print(val)      
                            deltaq = 1.0 - math.pow(val, mut_pow)

                        y = y + deltaq * (yu - yl)
                        #y = repair_solution_variable_value(y, yl, yu)  # Assuming this function is defined elsewhere
                        result[i][j] = y
            result[i] = self.repair(result[i], self.repair_method, self.lower_limits, self.upper_limits)
            #solution.variables()[i] = y
        return result

    def do_before(self, offspring: np.ndarray):
        """Conduct bounded polynomial mutation. Return the mutated individuals.

        Parameters
        ----------
        offspring : np.ndarray
            The array of offsprings to be mutated.

        Returns
        -------
        np.ndarray
            The mutated offsprings
        """
        min_val = np.ones_like(offspring) * self.lower_limits
        max_val = np.ones_like(offspring) * self.upper_limits
        k = np.random.random(offspring.shape)
        miu = np.random.random(offspring.shape)
        temp = np.logical_and((k <= self.ProM), (miu < 0.5))
        offspring_scaled = (offspring - min_val) / (max_val - min_val)
        offspring[temp] = offspring[temp] + (
            (max_val[temp] - min_val[temp])
            * (
                (
                    2 * miu[temp]
                    + (1 - 2 * miu[temp])
                    * (1 - offspring_scaled[temp]) ** (self.DisM + 1)
                )
                ** (1 / (self.DisM + 1))
                - 1
            )
        )
        temp = np.logical_and((k <= self.ProM), (miu >= 0.5))
        offspring[temp] = offspring[temp] + (
            (max_val[temp] - min_val[temp])
            * (
                1
                - (
                    2 * (1 - miu[temp])
                    + 2 * (miu[temp] - 0.5) * offspring_scaled[temp] ** (self.DisM + 1)
                )
                ** (1 / (self.DisM + 1))
            )
        )
        offspring[offspring > max_val] = max_val[offspring > max_val]
        offspring[offspring < min_val] = min_val[offspring < min_val]
        return offspring
