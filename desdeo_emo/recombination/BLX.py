import numpy as np
from desdeo_emo.recombination.CrossoverBase import CrossoverBase
from desdeo_emo.population.Population import Population

#from random import shuffle
#from numba import njit


class BLX(CrossoverBase):
    """Simulated binary crossover.

    Parameters
        ----------
        ProC : float, optional
            [description], by default 1
        DisC : float, optional
            [description], by default 30
    """

    def __init__(self, pop:Population, prob: float = 0.5, alpha: float = 0.5, repair_method: str = "rand"):
        """[summary]


        """
        self.pop = pop
        self.alpha = alpha
        self.prob = prob
        self.repair_method = repair_method

    def do(self, pop: np.ndarray, mating_pop_ids: list = None) -> np.ndarray:
        """Consecutive members of mating_pop_ids are crossed over
            in pairs. Example: if mating_pop_ids = [0, 2, 3, 6, 5] then the individuals
            are crossover as: [0, 2], [3, 6], [5, 0]. Note: if the number of elements
            is odd, the last individual is crossed over with the first one.

        Parameters
        ----------
        pop : np.ndarray
            Array of all individuals
        mating_pop_ids : list, optional
            Indices of population members to mate, by default None, which shuffles and
                mates whole population

        Returns
        -------
        np.ndarray
            The offspring produced as a result of crossover.
        """
        #lower_bounds = pop.problem.get_variable_lower_bounds()
        #upper_bounds = pop.problem.get_variable_upper_bounds()
        pop_size, num_var = pop.shape
        if mating_pop_ids is None:
            shuffled_ids = list(range(pop_size))
            np.random.shuffle(shuffled_ids)
        else:
            shuffled_ids = mating_pop_ids
        mating_pop = pop[shuffled_ids]
        mate_size = len(shuffled_ids)
        if len(shuffled_ids) % 2 == 1:
            # Maybe it should be pop_size-1?
            mating_pop = np.vstack((mating_pop, mating_pop[0]))
            mate_size = mate_size + 1
        offspring = np.zeros_like(mating_pop)  # empty_like() more efficient?
        for i in range(0, mate_size, 2):
            parent1 = mating_pop[i].copy()
            parent2 = mating_pop[i + 1].copy()

            offspring[i] = mating_pop[i].copy()
            offspring[i+1] = mating_pop[i + 1].copy()
            if (np.random.rand() <= self.prob):
                for j in range(len(parent1)):
                    value_x1 = parent1[j]
                    value_x2 = parent2[j]

                    if value_x2 > value_x1:
                        max_val = value_x2
                        min_val = value_x1
                    else:
                        max_val = value_x1
                        min_val = value_x2

                    range_val = max_val - min_val
                    min_range = min_val - range_val * self.alpha
                    max_range = max_val + range_val * self.alpha
                    
                    random_val = np.random.rand()
                    value_y1 = min_range + random_val * (max_range - min_range)

                    random_val = np.random.rand()
                    value_y2 = min_range + random_val * (max_range - min_range)

                    # Assuming solutionRepair.repairSolutionVariableValue is a function you've defined elsewhere
                    #value_y1 = self.repair(value_y1, self.repair_method, upper_bounds[j], lower_bounds[j])
                    #value_y2 = self.repair(value_y2, self.repair_method, upper_bounds[j], lower_bounds[j])

                    offspring[i][j] = value_y1
                    offspring[i + 1][j]  = value_y2

                offspring[i] = self.repair(offspring[i], self.repair_method)
                offspring[i + 1] = self.repair(offspring[i + 1], self.repair_method)

           
        return offspring

        
