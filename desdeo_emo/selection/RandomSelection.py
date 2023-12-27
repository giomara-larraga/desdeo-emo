import numpy as np
from typing import List
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.population.Population import Population


class RandomSelection(SelectionBase):
    """
    Random selection operator.

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
    def do(self, pop, fitness) -> List[int]:
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
        self.fitness = fitness
        self.pop_size = self.fitness.shape[0]
        parents = []
        for i in range(int(self.pop_size)):
            parents.append(
                np.asarray(
                    self._tour_select(),
                    self._tour_select(),
                )
            )
        return parents

    def fast_fill_random(X, N, columns=None, Xp=None, n_max_attempts=10):
        """

        Parameters
        ----------
        X : np.ndarray
            The actually array to fill with random values.
        N : int
            The upper limit for the values. The values will be in range (0, ..., N)
        columns : list
            The columns which should be filled randomly. Other columns indicate duplicates
        Xp : np.ndarray
            If some other duplicates shall be avoided by default

        """

        _, n_cols = X.shape

        if columns is None:
            columns = range(n_cols)

        # all columns set so far to be checked for duplicates
        J = []

        # for each of the columns which should be set to be no duplicates
        for col in columns:
            D = X[:, J]
            if Xp is not None:
                D = np.column_stack([D, Xp])

            # all the remaining indices that need to be filled with no duplicates
            rem = np.arange(len(X))

            for _ in range(n_max_attempts):
                if len(rem) > N:
                    X[rem, col] = np.random.choice(N, replace=True, size=len(rem))
                else:
                    X[rem, col] = np.random.permutation(N)[: len(rem)]

                rem = np.where((X[rem, col][:, None] == D[rem]).any(axis=1))[0]

                if len(rem) == 0:
                    break

            J.append(col)

        return X
