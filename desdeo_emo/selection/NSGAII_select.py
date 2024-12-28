import numpy as np

# from pygmo import fast_non_dominated_sorting as nds
from desdeo_tools.utilities import fast_non_dominated_sort
from typing import List
from desdeo_emo.selection.SelectionBase import InteractiveDominanceSelectionBase
from desdeo_emo.population.Population import Population


class NSGAII_select(InteractiveDominanceSelectionBase):
    """The NSGA-III selection operator. Code is heavily based on the version of nsga3 in
        the pymoo package by msu-coinlab.

    Parameters
    ----------
    pop : Population
        [description]
    n_survive : int, optional
        [description], by default None

    """

    def __init__(
        self, pop: Population, n_survive: int = None, selection_type: str = None
    ):
        super().__init__(pop.problem.n_of_fitnesses, selection_type)
        self.worst_fitness: np.ndarray = -np.full((1, pop.fitness.shape[1]), np.inf)
        self.extreme_points: np.ndarray = None
        if n_survive is None:
            self.n_survive: int = pop.pop_size
        if selection_type is None:
            selection_type = "mean"
        self.selection_type = selection_type
        self.ideal: np.ndarray = pop.ideal_fitness_val

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
        # ref_dirs = self.vectors.values_planar
        fitness = self._calculate_fitness(pop)
        # Calculating fronts and ranks
        # fronts, dl, dc, rank = nds(fitness)
        fronts = fast_non_dominated_sort(fitness)
        fronts = [np.where(fronts[i])[0] for i in range(len(fronts))]
        non_dominated = fronts[0]
        fmin = np.amin(fitness, axis=0)
        self.ideal = np.amin(np.vstack((self.ideal, fmin)), axis=0)

        # Calculating worst points
        self.worst_fitness = np.amax(np.vstack((self.worst_fitness, fitness)), axis=0)
        worst_of_population = np.amax(fitness, axis=0)
        worst_of_front = np.max(fitness[non_dominated, :], axis=0)
        self.extreme_points = self.get_extreme_points_c(
            fitness[non_dominated, :], self.ideal, extreme_points=self.extreme_points
        )
        nadir_point = self.get_nadir_point(
            self.extreme_points,
            self.ideal,
            self.worst_fitness,
            worst_of_population,
            worst_of_front,
        )

        # Finding individuals in first 'n' fronts
        selection = np.asarray([], dtype=int)
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
            crowding_distance = self.calc_crowding_distance(fitness[last_front, :])

            # if there is only one front
            if len(fronts) == 1:
                n_remaining = self.n_survive
                until_last_front = np.array([], dtype=np.int)

            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                n_remaining = self.n_survive - len(until_last_front)

            selected_from_last_front = np.argsort(crowding_distance)[::-1][:n_remaining]

            final_selection = np.concatenate(
                (until_last_front, last_front[selected_from_last_front])
            )

            # print(final_selection)
            if self.extreme_points is None:
                print("Error")
            if final_selection is None:
                print("Error")
        else:
            final_selection = selection
        return final_selection.astype(int)

    def get_extreme_points_c(self, F, ideal_point, extreme_points=None):
        """Taken from pymoo"""
        # calculate the asf which is used for the extreme point decomposition
        asf = np.eye(F.shape[1])
        asf[asf == 0] = 1e6

        # add the old extreme points to never loose them for normalization
        _F = F
        if extreme_points is not None:
            _F = np.concatenate([extreme_points, _F], axis=0)

        # use __F because we substitute small values to be 0
        __F = _F - ideal_point
        __F[__F < 1e-3] = 0

        # update the extreme points for the normalization having the highest asf value
        # each
        F_asf = np.max(__F * asf[:, None, :], axis=2)
        I = np.argmin(F_asf, axis=1)
        extreme_points = _F[I, :]
        return extreme_points

    def get_nadir_point(
        self,
        extreme_points,
        ideal_point,
        worst_point,
        worst_of_front,
        worst_of_population,
    ):
        LinAlgError = np.linalg.LinAlgError
        try:
            # find the intercepts using gaussian elimination
            M = extreme_points - ideal_point
            b = np.ones(extreme_points.shape[1])
            plane = np.linalg.solve(M, b)
            intercepts = 1 / plane

            nadir_point = ideal_point + intercepts

            if (
                not np.allclose(np.dot(M, plane), b)
                or np.any(intercepts <= 1e-6)
                or np.any(nadir_point > worst_point)
            ):
                raise LinAlgError()

        except LinAlgError:
            nadir_point = worst_of_front

        b = nadir_point - ideal_point <= 1e-6
        nadir_point[b] = worst_of_population[b]
        return nadir_point

    def calc_crowding_distance(self, F):
        n_points, n_obj = F.shape

        # sort each column and get index
        I = np.argsort(F, axis=0, kind="mergesort")

        # sort the objective space values for the whole matrix
        F = F[I, np.arange(n_obj)]

        # calculate the distance from each point to the last and next
        dist = np.row_stack([F, np.full(n_obj, np.inf)]) - np.row_stack(
            [np.full(n_obj, -np.inf), F]
        )

        # calculate the norm for each objective - set to NaN if all values are equal
        norm = np.max(F, axis=0) - np.min(F, axis=0)
        norm[norm == 0] = np.nan

        # prepare the distance to last and next vectors
        dist_to_last, dist_to_next = dist, np.copy(dist)
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divide by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        cd = (
            np.sum(
                dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)],
                axis=1,
            )
            / n_obj
        )

        return cd

    def _calculate_fitness(self, pop) -> np.ndarray:
        if self.selection_type == "mean":
            return pop.fitness
        if self.selection_type == "optimistic":
            return pop.fitness - pop.uncertainity
        if self.selection_type == "robust":
            return pop.fitness + pop.uncertainity
