import numpy as np
from typing import List
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.population.Population import Population


class NTournamentSelection(SelectionBase):
    """
    Tournament selection operator.

    Parameters
    ----------
    pop : Population
        The population of individuals
    tournament_size : int
        Size of the tournament.
    """

    def __init__(self, pop, tournament_size):
        # initialize
        # self.fitness = pop.fitness
        # self.pop_size = pop.pop_size
        self.tournament_size = tournament_size

    # TODO: add the opt fitness, which is the custom set of fitness values e.g self.local_fitnesss
    def do(self, pop) -> List[int]:
        """Performs tournament selections and returns the parents.
        Parameters
        ----------
        pop : Population
            The current population.

        Returns
        -------
        List[int]
            List of indices of the selected individuals
        """
        fitness = pop.fitness
        parents = []
        seq = np.arange(fitness.shape[0])

        for i in range(fitness.shape[0]):
            selected = np.random.choice(seq, size = self.tournament_size, replace=False)
            parents.append(self.find_index_of_best_solution(fitness, selected))
            return

        return parents

    def find_index_of_best_solution(self, fitness, solution_list: np.array):
        index = 0
        best_known = solution_list[0]

        for i in range(1, len(solution_list)):
            candidate_solution = solution_list[i]
            flag = self.comparator(fitness[best_known], fitness[candidate_solution])
            if flag == 1:
                index = i
                best_known = candidate_solution

        return index

    def comparator(self, vector1, vector2):
        best_is_one = 0
        best_is_two = 0

        for i in range(len(vector1)):
            value1 = vector1[i]
            value2 = vector2[i]

            if value1 != value2:
                if value1 < value2:
                    best_is_one = 1
                if value2 < value1:
                    best_is_two = 1

        result = 1 if best_is_two > best_is_one else (-1 if best_is_two < best_is_one else 0)
        return result

