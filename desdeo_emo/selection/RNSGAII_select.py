from desdeo_tools.interaction import ReferencePointPreference
import numpy as np

# from pygmo import fast_non_dominated_sorting as nds
from desdeo_tools.utilities import fast_non_dominated_sort
from typing import List
from desdeo_emo.selection.SelectionBase import InteractiveDominanceSelectionBase
from desdeo_emo.population.Population import Population


class RNSGAII_select(InteractiveDominanceSelectionBase):
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
        self,
        pop: Population,
        n_survive: int = None,
        selection_type: str = None,
        epsilon=0.001,
        normalization="front",
        weights=None,
        extreme_points_as_reference_points=False,
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
        self.epsilon = epsilon
        self.normalization = normalization
        self.weights = weights
        self.n_obj = pop.objectives.shape[1]
        if self.weights is None:
            self.weights = np.full(self.n_obj, 1 / self.n_obj)
        self.extreme_points_as_reference_points = extreme_points_as_reference_points
        self.ref_points = None
        self.ideal_vector = np.full(self.n_obj, np.inf)
        self.nadir_vector = np.full(self.n_obj, -np.inf)

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
        # the final indices of surviving individuals
        survivors = []

        F = self._calculate_fitness(pop)
        # Calculating fronts and ranks
        # fronts, dl, dc, rank = nds(fitness)
        fronts = fast_non_dominated_sort(F)
        fronts = [np.where(fronts[i])[0] for i in range(len(fronts))]

        if self.normalization == "ever":
            # find or usually update the new ideal point - from feasible solutions
            self.ideal_vector = np.min(np.vstack((self.ideal_vector, F)), axis=0)
            self.nadir_vector = np.max(np.vstack((self.nadir_vector, F)), axis=0)

        elif self.normalization == "front":
            front = fronts[0]
            
            if len(front) > 1:
                self.ideal_vector = np.min(F[front], axis=0)
                self.nadir_vector = np.max(F[front], axis=0)
    

        elif self.normalization == "no":
            self.ideal_vector = np.zeros(self.n_obj)
            self.nadir_vector = np.ones(self.n_obj)

        if self.extreme_points_as_reference_points:
            self.ref_points = np.row_stack(
                [self.ref_points, self.get_extreme_points_c(F, self.ideal_vector)]
            )

        # calculate the distance matrix from ever solution to all reference point
        dist_to_ref_points = self.calc_norm_pref_distance(
            F, self.ref_points, self.weights, self.ideal_vector, self.nadir_vector
        )

        for k, front in enumerate(fronts):
            # save rank attributes to the individuals - rank = front here
            # pop[front].set("rank", np.full(len(front), k))

            # number of individuals remaining
            n_remaining = self.n_survive - len(survivors)

            # the ranking of each point regarding each reference point (two times argsort is necessary)
            rank_by_distance = np.argsort(
                np.argsort(dist_to_ref_points[front], axis=0), axis=0
            )

            # the reference point where the best ranking is coming from
            ref_point_of_best_rank = np.argmin(rank_by_distance, axis=1)

            # the actual ranking which is used as crowding
            ranking = rank_by_distance[np.arange(len(front)), ref_point_of_best_rank]

            if len(front) <= n_remaining:
                # we can simply copy the crowding to ranking. not epsilon selection here
                crowding = ranking
                I = np.arange(len(front))

            else:
                # Distance from solution to every other solution and set distance to itself to infinity
                dist_to_others = self.calc_norm_pref_distance(
                    F[front], F[front], self.weights, self.ideal_vector, self.nadir_vector
                )
                np.fill_diagonal(dist_to_others, np.inf)

                # the crowding that will be used for selection
                crowding = np.full(len(front), np.nan)

                # solutions which are not already selected - for
                not_selected = np.argsort(ranking)

                # until we have saved a crowding for each solution
                while len(not_selected) > 0:
                    # select the closest solution
                    idx = not_selected[0]

                    # set crowding for that individual
                    crowding[idx] = ranking[idx]

                    # need to remove myself from not-selected array
                    to_remove = [idx]

                    # Group of close solutions
                    dist = dist_to_others[idx][not_selected]
                    group = not_selected[np.where(dist < self.epsilon)[0]]

                    # if there exists solution with a distance less than epsilon
                    if len(group):
                        # discourage them by giving them a high crowding
                        crowding[group] = ranking[group] + np.round(len(front) / 2)

                        # remove group from not_selected array
                        to_remove.extend(group)

                    not_selected = np.array(
                        [i for i in not_selected if i not in to_remove]
                    )

                # now sort by the crowding (actually modified rank) ascending and let the best survive
                I = np.argsort(crowding)[:n_remaining]

            # set the crowding to all individuals
            # pop[front].set("crowding", crowding)

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        # inverse of crowding because nsga2 does maximize it (then tournament selection can stay the same)
        # pop.set("crowding", -pop.get("crowding"))

        return survivors

    def get_extreme_points_c(self, F, ideal_vector, extreme_points=None):
        """Taken from pymoo"""
        # calculate the asf which is used for the extreme point decomposition
        asf = np.eye(F.shape[1])
        asf[asf == 0] = 1e6

        # add the old extreme points to never loose them for normalization
        _F = F
        if extreme_points is not None:
            _F = np.concatenate([extreme_points, _F], axis=0)

        # use __F because we substitute small values to be 0
        __F = _F - ideal_vector
        __F[__F < 1e-3] = 0

        # update the extreme points for the normalization having the highest asf value
        # each
        F_asf = np.max(__F * asf[:, None, :], axis=2)
        I = np.argmin(F_asf, axis=1)
        extreme_points = _F[I, :]
        return extreme_points

    def get_nadir_vector(
        self,
        extreme_points,
        ideal_vector,
        worst_point,
        worst_of_front,
        worst_of_population,
    ):
        LinAlgError = np.linalg.LinAlgError
        try:
            # find the intercepts using gaussian elimination
            M = extreme_points - ideal_vector
            b = np.ones(extreme_points.shape[1])
            plane = np.linalg.solve(M, b)
            intercepts = 1 / plane

            nadir_vector = ideal_vector + intercepts

            if (
                not np.allclose(np.dot(M, plane), b)
                or np.any(intercepts <= 1e-6)
                or np.any(nadir_vector > worst_point)
            ):
                raise LinAlgError()

        except LinAlgError:
            nadir_vector = worst_of_front

        b = nadir_vector - ideal_vector <= 1e-6
        nadir_vector[b] = worst_of_population[b]
        return nadir_vector

    def calc_norm_pref_distance(self, A, B, weights, ideal, nadir):
        D = np.repeat(A, B.shape[0], axis=0) - np.tile(B, (A.shape[0], 1))
        denominators = nadir - ideal
        if np.any(np.abs(denominators) < np.finfo(float).eps):
            # Reshape A and B to have the same dimensions for broadcasting
            A_reshaped = A[:, np.newaxis, :]
            B_reshaped = B[np.newaxis, :, :]
            N= np.linalg.norm(A_reshaped - B_reshaped, axis=2)
        else:
            #denominators[denominators == 0] = np.finfo(float).eps
            N = ((D / denominators) ** 2) * weights #Check division by zero here
            N = np.sqrt(np.sum(N, axis=1) * len(weights))
        return np.reshape(N, (A.shape[0], B.shape[0]))

    def _calculate_fitness(self, pop) -> np.ndarray:
        if self.selection_type == "mean":
            return pop.fitness
        if self.selection_type == "optimistic":
            return pop.fitness - pop.uncertainity
        if self.selection_type == "robust":
            return pop.fitness + pop.uncertainity

    def manage_reference_point(
        self, pop: Population, preference: ReferencePointPreference
    ):
        if not isinstance(preference, ReferencePointPreference):
            raise TypeError(
                "Preference object must be an instance of ReferencePointPreference."
            )
        self.ref_points = preference.response.values * pop.problem._max_multiplier
