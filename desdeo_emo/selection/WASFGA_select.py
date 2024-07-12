from locale import normalize
import numpy as np

# from pygmo import fast_non_dominated_sorting as nds
from desdeo_tools.utilities import fast_non_dominated_sort
from typing import List,Union
from desdeo_emo.selection.SelectionBase import InteractiveDecompositionSelectionBase
from desdeo_emo.population.Population import Population
from desdeo_tools.scalarization.ASF import ASFBase
from desdeo_emo.utilities.WASFGA_ranking import WASFGARanking
from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors


class WASFGASF(ASFBase):
    def __init__(
        self,
        nadir: np.array,
        ideal: np.array,
        rho: float = 1e-6,

    ):
        self.nadir = nadir
        self.ideal =ideal
        self.rho = rho
    def __call__(self, objective_vector: np.ndarray, reference_point: np.ndarray, reference_vector: np.ndarray) -> Union[float, np.ndarray]:
        # assure this function works with single objective vectors
        if objective_vector.ndim == 1:
            objective_vector = objective_vector.reshape((1, -1))
        diff=  (objective_vector - reference_point)
        
        max_term = np.max(reference_vector * diff)
        sum_term = self.rho * np.sum(reference_vector * diff)

        return max_term + sum_term

class WASFGA_select(InteractiveDecompositionSelectionBase):
    """The WASFGA selection operator. 

    Parameters
    ----------
    pop : Population
        [description]
    n_survive : int, optional
        [description], by default None

    """

    def __init__(
        self, pop: Population, reference_point: np.array, n_survive: int = None, selection_type: str = None
        ):
        super().__init__(pop.pop_size, pop.problem.n_of_fitnesses, selection_type)
        if selection_type is None:
            self.selection_type = "mean"
        if n_survive is None:
            self.n_survive: int = pop.pop_size
        self.ideal: np.ndarray = pop.ideal_fitness_val
        self.nadir: np.ndarray = pop.nadir_fitness_val
        self.reference_point: np.ndarray = reference_point
        self.scalarization_function: ASFBase = WASFGASF(self.nadir, self.ideal)
        self.ranking_method: WASFGARanking = WASFGARanking(self.scalarization_function)
        self.vectors.invert_WASFGA(True)
        


    def do(self, pop: Population) -> List[int]:
        """Select individuals for mating for NSGA-III.

        Parameters
        ----------
        pop : Population
            The current population.

        Returns
        -------
        List[int]
            List of indices of the selected individuals
        """

        fitness = self._calculate_fitness(pop)

        # Calculating fronts and ranks
        fronts = self._compute_fronts(pop, fitness)

        fronts = [np.where(fronts[i])[0] for i in range(len(fronts))]
        non_dominated = fronts[0]

        # Finding individuals in first 'n' fronts
        selection = np.asarray([], dtype=int)
        
        #while (len(selection) < self.n_survive):
        for front_id in range(len(fronts)):
            if len(np.concatenate(fronts[: front_id + 1])) < self.n_survive:
                continue
            else:
                fronts = fronts[: front_id + 1]
                selection = np.concatenate(fronts)
                break
        F = fitness[selection]

        last_front = fronts[-1]

        # Selecting individuals from the last acceptable front.
        if len(selection) > self.n_survive:
            #sf_of_individuals = self.ranking_method.sf_values[selection]
            #n_remaining = len(selection) - self.n_survive

            # if there is only one front
            if len(fronts) == 1:
                n_remaining = self.n_survive
                until_last_front = np.array([], dtype=np.int)
    
            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                #id_until_last_front = list(range(len(until_last_front)))
                n_remaining = self.n_survive - len(until_last_front)


            #selected_from_last_front = last_front[0:n_remaining-1,:]
            final_selection = np.concatenate(
                (until_last_front, last_front[0:n_remaining-1])
            )
        else:
            final_selection = selection
        return final_selection.astype(int)

    def _calculate_fitness(self, pop) -> np.ndarray:
        if self.selection_type == "mean":
            return pop.fitness
        if self.selection_type == "optimistic":
            return pop.fitness - pop.uncertainity
        if self.selection_type == "robust":
            return pop.fitness + pop.uncertainity


    def _compute_fronts(self, pop, fitness) -> np.ndarray:
        num_solutions = pop.pop_size
        num_vectors = self.vectors.number_of_vectors
        self.sf_values = np.zeros((num_solutions, num_vectors))

        for i in range(0, num_solutions):
            for j in range(0, num_vectors):
                self.sf_values[i][j] = self.scalarization_function.__call__(pop.fitness[i], self.reference_point, self.referenceVectors.values[j])
        
        #print(self.sf_values)
        return fast_non_dominated_sort(self.sf_values)



    
