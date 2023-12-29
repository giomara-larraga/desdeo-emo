import numpy as np
from typing import List
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.population.Population import Population


class NRandomSelection(SelectionBase):
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
        seq = np.arange(fitness.shape[0])

        parents = np.random.choice(seq, size=len(seq), replace=False)

        return parents




