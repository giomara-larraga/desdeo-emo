import numpy as np

# from pygmo import fast_non_dominated_sorting as nds
from desdeo_tools.utilities import fast_non_dominated_sort
from typing import List
from desdeo_emo.selection.SelectionBase import InteractiveDecompositionSelectionBase
from desdeo_emo.population.Population import Population
from desdeo_tools.interaction import (
    ReferencePointPreference,
)


class RNSGAIII_select(InteractiveDecompositionSelectionBase):
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
        mu: float = None,
    ):
        super().__init__(pop.pop_size, pop.problem.n_of_fitnesses, selection_type)
        self.worst_fitness: np.ndarray = -np.full((1, pop.fitness.shape[1]), np.inf)
        self.extreme_points: np.ndarray = None
        if n_survive is None:
            self.n_survive: int = pop.pop_size
        if selection_type is None:
            selection_type = "mean"
        self.selection_type = selection_type
        self.ideal: np.ndarray = pop.ideal_fitness_val
        self.reference_point = None
        self.mu = mu

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

        aspiration_ref_dirs = self.vectors.values_planar

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
        unit_ref_points = (self.reference_point - self.ideal) / (
            nadir_point - self.ideal
        )
        # print(unit_ref_points)
        ref_dirs = self.get_ref_dirs_from_points(
            unit_ref_points, aspiration_ref_dirs, mu=self.mu
        )
        #print(ref_dirs)
        # Selecting individuals from the last acceptable front.
        if len(selection) > self.n_survive:
            niche_of_individuals, dist_to_niche = self.associate_to_niches(
                F, ref_dirs, self.ideal, nadir_point
            )
            # if there is only one front
            if len(fronts) == 1:
                n_remaining = self.n_survive
                until_last_front = np.array([], dtype=np.int)
                niche_count = np.zeros(len(ref_dirs), dtype=np.int)

            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                id_until_last_front = list(range(len(until_last_front)))
                niche_count = self.calc_niche_count(
                    len(ref_dirs), niche_of_individuals[id_until_last_front]
                )
                n_remaining = self.n_survive - len(until_last_front)

            last_front_selection_id = list(range(len(until_last_front), len(selection)))
            if np.any(selection[last_front_selection_id] != last_front):
                print("error!!!")
            selected_from_last_front = self.niching(
                fitness[last_front, :],
                n_remaining,
                niche_count,
                niche_of_individuals[last_front_selection_id],
                dist_to_niche[last_front_selection_id],
            )
            final_selection = np.concatenate(
                (until_last_front, last_front[selected_from_last_front])
            )
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

    def niching(self, F, n_remaining, niche_count, niche_of_individuals, dist_to_niche):
        survivors = []

        # boolean array of elements that are considered for each iteration
        mask = np.full(F.shape[0], True)

        while len(survivors) < n_remaining:
            # all niches where new individuals can be assigned to
            next_niches_list = np.unique(niche_of_individuals[mask])

            # pick a niche with minimum assigned individuals - break tie if necessary
            next_niche_count = niche_count[next_niches_list]
            next_niche = np.where(next_niche_count == next_niche_count.min())[0]
            next_niche = next_niches_list[next_niche]
            next_niche = next_niche[np.random.randint(0, len(next_niche))]

            # indices of individuals that are considered and assign to next_niche
            next_ind = np.where(
                np.logical_and(niche_of_individuals == next_niche, mask)
            )[0]

            # shuffle to break random tie (equal perp. dist) or select randomly
            np.random.shuffle(next_ind)

            if niche_count[next_niche] == 0:
                next_ind = next_ind[np.argmin(dist_to_niche[next_ind])]
            else:
                # already randomized through shuffling
                next_ind = next_ind[0]

            mask[next_ind] = False
            survivors.append(int(next_ind))

            niche_count[next_niche] += 1

        return survivors

    def associate_to_niches(
        self, F, ref_dirs, ideal_point, nadir_point, utopian_epsilon=0.0
    ):
        utopian_point = ideal_point - utopian_epsilon

        denom = nadir_point - utopian_point
        denom[denom == 0] = 1e-12

        # normalize by ideal point and intercepts
        N = (F - utopian_point) / denom
        dist_matrix = self.calc_perpendicular_distance(N, ref_dirs)

        niche_of_individuals = np.argmin(dist_matrix, axis=1)
        dist_to_niche = dist_matrix[np.arange(F.shape[0]), niche_of_individuals]

        return niche_of_individuals, dist_to_niche

    def calc_niche_count(self, n_niches, niche_of_individuals):
        niche_count = np.zeros(n_niches, dtype=np.int)
        index, count = np.unique(niche_of_individuals, return_counts=True)
        niche_count[index] = count
        return niche_count

    def calc_perpendicular_distance(self, N, ref_dirs):
        u = np.tile(ref_dirs, (len(N), 1))
        v = np.repeat(N, len(ref_dirs), axis=0)

        norm_u = np.linalg.norm(u, axis=1)

        scalar_proj = np.sum(v * u, axis=1) / norm_u
        proj = scalar_proj[:, None] * u / norm_u[:, None]
        val = np.linalg.norm(proj - v, axis=1)
        matrix = np.reshape(val, (len(N), len(ref_dirs)))

        return matrix

    def _calculate_fitness(self, pop) -> np.ndarray:
        if self.selection_type == "mean":
            return pop.fitness
        if self.selection_type == "optimistic":
            return pop.fitness - pop.uncertainity
        if self.selection_type == "robust":
            return pop.fitness + pop.uncertainity

    def get_ref_dirs_from_points(self, ref_point, ref_dirs, mu=0.1):
        """
        This function takes user specified reference points, and creates smaller sets of equidistant
        Das-Dennis points around the projection of user points on the Das-Dennis hyperplane
        :param ref_point: List of user specified reference points
        :param n_obj: Number of objectives to consider
        :param mu: Shrinkage factor (0-1), Smaller = tigher convergence, Larger= larger convergence
        :return: Set of reference points
        """

        n_obj = ref_point.shape[1]

        val = []
        n_vector = np.ones(n_obj) / np.sqrt(n_obj)  # Normal vector of Das Dennis plane
        point_on_plane = np.eye(n_obj)[0]  # Point on Das-Dennis

        for point in ref_point:
            ref_dir_for_aspiration_point = np.copy(
                ref_dirs
            )  # Copy of computed reference directions
            ref_dir_for_aspiration_point = mu * ref_dir_for_aspiration_point

            cent = np.mean(
                ref_dir_for_aspiration_point, axis=0
            )  # Find centroid of shrunken reference points

            # Project shrunken Das-Dennis points back onto original Das-Dennis hyperplane
            intercept = self.line_plane_intersection(
                np.zeros(n_obj), point, point_on_plane, n_vector
            )
            shift = intercept - cent  # shift vector

            ref_dir_for_aspiration_point += shift

            # If reference directions are located outside of first octant, redefine points onto the border
            if not (ref_dir_for_aspiration_point > 0).min():
                ref_dir_for_aspiration_point[ref_dir_for_aspiration_point < 0] = 0
                ref_dir_for_aspiration_point = (
                    ref_dir_for_aspiration_point
                    / np.sum(ref_dir_for_aspiration_point, axis=1)[:, None]
                )
            val.extend(ref_dir_for_aspiration_point)

        val.extend(np.eye(n_obj))  # Add extreme points
        return np.array(val)

    # intersection function

    def line_plane_intersection(self, l0, l1, p0, p_no, epsilon=1e-6):
        """
        l0, l1: define the line
        p0, p_no: define the plane:
            p0 is a point on the plane (plane coordinate).
            p_no is a normal vector defining the plane direction;
                (does not need to be normalized).

        reference: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
        return a Vector or None (when the intersection can't be found).
        """

        l = l1 - l0
        dot = np.dot(l, p_no)

        if abs(dot) > epsilon:
            # the factor of the point between p0 -> p1 (0 - 1)
            # if 'fac' is between (0 - 1) the point intersects with the segment.
            # otherwise:
            #  < 0.0: behind p0.
            #  > 1.0: infront of p1.
            w = p0 - l0
            d = np.dot(w, p_no) / dot
            l = l * d
            return l0 + l
        else:
            # The segment is parallel to plane then return the perpendicular projection
            ref_proj = l1 - (np.dot(l1 - p0, p_no) * p_no)
            return ref_proj

    def manage_reference_point(
        self, pop: Population, preference: ReferencePointPreference
    ):
        if not isinstance(preference, ReferencePointPreference):
            raise TypeError(
                "Preference object must be an instance of ReferencePointPreference."
            )
        ideal = pop.ideal_fitness_val
        self.reference_point = preference.response.values * pop.problem._max_multiplier
        # refpoint = refpoint - ideal
        # norm = np.sqrt(np.sum(np.square(refpoint)))
        # refpoint = refpoint / norm
        # self.vectors.iteractive_adapt_3(refpoint)
        # self.vectors.add_edge_vectors()
        # self.vectors.neighbouring_angles()
