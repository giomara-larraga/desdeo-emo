from typing import Dict

import numpy as np
from scipy.spatial.distance import cdist as s_cdist

from numpy.random import permutation
from scipy.spatial import distance_matrix
from desdeo_emo.EAs.BaseEA import BaseDecompositionEA
from desdeo_emo.population.Population import Population
from desdeo_problem import MOProblem
from desdeo_emo.utilities import ReferenceVectors
from desdeo_emo.selection.TournamentSelection import TournamentSelection
from desdeo_emo.selection.MOEAD_select import MOEAD_select
from desdeo_emo.recombination.BoundedPolynomialMutation import BP_mutation
from desdeo_emo.recombination.SimulatedBinaryCrossover import SBX_xover

from desdeo_tools.scalarization import MOEADSF
from desdeo_tools.scalarization.MOEADSF import Tchebycheff, PBI
from desdeo_problem import MOProblem, classificationPISProblem
from desdeo_tools.interaction import (
    SimplePlotRequest,
    ReferencePointPreference,
    PreferredSolutionPreference,
    NonPreferredSolutionPreference,
    BoundPreference,
    validate_ref_point_data_type,
    validate_ref_point_dimensions,
    validate_ref_point_with_ideal,
)

class IMOEA_D(BaseDecompositionEA):
    """Python implementation of MOEA/D

    .. Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition," 
    in IEEE Transactions on Evolutionary Computation, vol. 11, no. 6, pp. 712-731, Dec. 2007, doi: 10.1109/TEVC.2007.892759.

    Parameters
    ----------
    problem: MOProblem
    	The problem class object specifying the details of the problem.
    scalarization_function: MOEADSF
    	The scalarization function to compare the solutions. Some implementations 
        can be found in desdeo-tools/scalarization/MOEADSF. By default it uses the
        PBI function.
    n_neighbors: int, optional
    	Number of reference vectors considered in the neighborhoods creation. The default 
    	number is 20.
    population_params: Dict, optional
    	The parameters for the population class, by default None. See
        desdeo_emo.population.Population for more details.
    initial_population: Population, optional
    	An initial population class, by default None. Use this if you want to set up
        a specific starting population, such as when the output of one EA is to be
        used as the input of another.
    lattice_resolution: int, optional
    	The number of divisions along individual axes in the objective space to be
        used while creating the reference vector lattice by the simplex lattice
        design. By default None
    n_parents: int, optional
    	Number of individuals considered for the generation of offspring solutions. The default
    	option is 2.
    a_priori: bool, optional
    	A bool variable defining whether a priori preference is to be used or not.
        By default False
    interact: bool, optional
    	A bool variable defining whether interactive preference is to be used or
        not. By default False
    use_surrogates: bool, optional
    	A bool variable defining whether surrogate problems are to be used or
        not. By default False
    n_iterations: int, optional
     	The total number of iterations to be run, by default 10. This is not a hard
        limit and is only used for an internal counter.
    n_gen_per_iter: int, optional
    	The total number of generations in an iteration to be run, by default 100.
        This is not a hard limit and is only used for an internal counter.
    total_function_evaluations: int, optional
    	Set an upper limit to the total number of function evaluations. When set to
        zero, this argument is ignored and other termination criteria are used.
    """

    def __init__(  # parameters of the class
        self,
        problem: MOProblem,
        scalarization_function: MOEADSF = PBI(),
        n_neighbors: int = 20,
        population_params: Dict = None,
        initial_population: Population = None,
        lattice_resolution: int = None,
        use_repair: bool = True,
        n_parents: int = 2,
        a_priori: bool = False,
        interact: bool = False,
        use_surrogates: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        predefined_distance = 0.2,
    ):
        super().__init__(  # parameters for decomposition based approach
            problem=problem,
            population_size=None,
            population_params=population_params,
            initial_population=initial_population,
            lattice_resolution=lattice_resolution,
            a_priori=a_priori,
            interact=interact,
            use_surrogates=use_surrogates,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
        )
        self.population_size = self.reference_vectors.number_of_vectors
        self.problem = problem
        self.scalarization_function = scalarization_function
        self.n_neighbors = n_neighbors

        self.use_repair = use_repair
        self.n_parents = n_parents
        self.predefined_distance = predefined_distance

        selection_operator = MOEAD_select(
            self.population, SF_type=self.scalarization_function
        )
        self.selection_operator = selection_operator
        # Compute the distance between each pair of reference vectors
        distance_matrix_vectors = distance_matrix(
            self.reference_vectors.values_planar, self.reference_vectors.values_planar
        )
        # Get the closest vectors to obtain the neighborhoods
        self.neighborhoods = np.argsort(
            distance_matrix_vectors, axis=1, kind="quicksort"
        )[:, :n_neighbors]
        self.population.update_ideal()
        self._ideal_point = np.copy(self.population.ideal_fitness_val)

    def _next_gen(self):
        # For each individual from the population
        for i in range(self.population_size):
            # Consider only the individuals of the current neighborhood
            # for parent selection
            current_neighborhood = self.neighborhoods[i, :]
            
            selected_parents = current_neighborhood[permutation(self.n_neighbors)][
                : self.n_parents
            ]

            # Apply genetic operators over two random individuals
            offspring = self.population.mate(selected_parents)
            offspring = offspring[0, :]

            # Repair the solution if it is needed
            if self.use_repair:
                offspring = self.population.repair(offspring)

            # Evaluate the offspring using the objective function
            results_off = self.problem.evaluate(offspring, self.use_surrogates)

            offspring_fx = results_off.fitness[0, :]

            self._function_evaluation_count += 1

            # Update the ideal point
            self._ideal_point = np.min(
                np.vstack([self._ideal_point, offspring_fx]), axis=0
            )

            # Replace individuals with a worse SF value than the offspring
            selected = self._select(current_neighborhood, offspring_fx)

            self.population.replace(selected, offspring, results_off)
        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1

    def set_spread_parameter(self, new_distance):
        self.predefined_distance = new_distance
    def manage_preferences(self, preference=None):
            """Run the interruption phase of EA.

            Use this phase to make changes to RVEA.params or other objects.
            Updates Reference Vectors (adaptation), conducts interaction with the user.
            """
            #print("preferencias desde MOEA/D")
            if isinstance(preference, Dict) and isinstance(
                self.population.problem, classificationPISProblem
            ):
                self.population.problem.update_preference(preference)
                self.population.reevaluate_fitness()
                self.reference_vectors.adapt(self.population.fitness)
                return

            if not isinstance(
                preference,
                (
                    ReferencePointPreference,
                    PreferredSolutionPreference,
                    NonPreferredSolutionPreference,
                    BoundPreference,
                    Dict,
                    type(None),
                ),
            ):
                msg = (
                    f"Wrong object sent as preference. Expected type = "
                    f"{type(ReferencePointPreference)}\n"
                    f"{type(PreferredSolutionPreference)}\n"
                    f"{type(NonPreferredSolutionPreference)}\n"
                    f"{type(BoundPreference)}, Dict, or None\n"
                    f"Recieved type = {type(preference)}"
                )
                raise self.eaError(msg)

            if preference is not None:
                if preference.request_id != self._interaction_request_id:
                    msg = (
                        f"Wrong preference object sent. Expected id = "
                        f"{self._interaction_request_id}.\n"
                        f"Recieved id = {preference.request_id}"
                    )
                    raise self.eaError(msg)
            if preference is None and not self._ref_vectors_are_focused:
                self.reference_vectors.adapt(self.population.fitness)
            if isinstance(preference, ReferencePointPreference):
                #print("reference point")
                ideal = self.population.ideal_fitness_val
                refpoint = (
                    preference.response.values * self.population.problem._max_multiplier
                )
                refpoint = refpoint - ideal
                norm = np.sqrt(np.sum(np.square(refpoint)))
                refpoint = refpoint / norm

                filter_rp = preference.response.values * self.population.problem._max_multiplier

                # Check which solutions will remain in the population and how many will be repositioned
                solutions_remain = self.reference_vectors.check_roi_rp(filter_rp, self.predefined_distance)
                new_solutions = self.reference_vectors.number_of_vectors - np.count_nonzero(solutions_remain)
                #print (solutions_remain)


                #Replace the reference vectors that are not in the ROI
                #self.reference_vectors.values[solutions_remain==False] = np.vstack([reference_vectors_3.values[solutions_remain], new_reference_vectors_3.values])
                #self.reference_vectors.values_planar[solutions_remain==False] = np.vstack([reference_vectors_3.values_planar[solutions_remain], new_reference_vectors_3.values_planar])

                if (new_solutions>0):
                    # Create a new set of weight vectors and adapt them
                    new_reference_vectors = ReferenceVectors(number_of_objectives=self.problem.n_of_objectives,  approx_number = new_solutions, creation_type="Approximated")
                    #print(new_reference_vectors.values)
                    new_reference_vectors.reposition_RVs(refpoint, self.predefined_distance)

                    #indexes_new = np.invert(solutions_remain)
                    
                    to_adapt = np.random.permutation(new_solutions)

                    self.reference_vectors.values[np.invert(solutions_remain),:] = new_reference_vectors.values[to_adapt,:]
                    self.reference_vectors.values_planar[np.invert(solutions_remain),:] = new_reference_vectors.values_planar[to_adapt,:]
                    #self.reference_vectors.normalize()
                    #self.reference_vectors.add_edge_vectors_position()
                
                #print(self.reference_vectors.values)
            elif isinstance(preference, PreferredSolutionPreference):
                #self.reference_vectors.interactive_adapt_1(
                #    z=self.population.objectives[preference.response],
                #    n_solutions=np.shape(self.population.objectives)[0],
                #)
                #self.reference_vectors.add_edge_vectors()
                
                # Check which solutions will remain in the population and how many will be repositioned
                ref_vector_selected = self.reference_vectors.values_planar[preference.response]
                solutions_remain = self.reference_vectors.check_roi_ps(ref_vector_selected, self.predefined_distance)
                new_solutions = self.reference_vectors.number_of_vectors - np.count_nonzero(solutions_remain)
                #print (solutions_remain)

                # Create a new set of weight vectors and adapt them
                if (new_solutions>0):
                    new_reference_vectors = ReferenceVectors(number_of_objectives=self.problem.n_of_objectives,  approx_number = new_solutions, creation_type="Approximated")
                    new_reference_vectors.reposition_RVs(ref_vector_selected, self.predefined_distance)
                    to_adapt = np.random.permutation(new_solutions)

                if (new_solutions>0):
                    self.reference_vectors.values[np.invert(solutions_remain),:] = new_reference_vectors.values[to_adapt,:]
                    self.reference_vectors.values_planar[np.invert(solutions_remain),:] = new_reference_vectors.values_planar[to_adapt,:]

            elif isinstance(preference, NonPreferredSolutionPreference):
                ref_vector_selected = self.reference_vectors.values_planar[preference.response]

                self.reference_vectors.interactive_adapt_2(
                    z=self.population.objectives[preference.response],
                    n_solutions=np.shape(self.population.objectives)[0],
                )
                
            elif isinstance(preference, BoundPreference):
                preferred_ranges = preference.response
                #print(preferred_ranges)
                #self.reference_vectors.interactive_adapt_4(preference.response)
                
                solutions_remain = self.check_roi_ranges(preferred_ranges)
                #print(solutions_remain)
                #print(np.count_nonzero(solutions_remain))
                #print(self.reference_vectors.number_of_vectors)
                new_solutions = self.reference_vectors.number_of_vectors - np.count_nonzero(solutions_remain)
            

                # Create a new set of weight vectors and adapt them
                
                new_reference_vectors = ReferenceVectors(number_of_objectives=self.problem.n_of_objectives,  approx_number = new_solutions, creation_type="Approximated")
                #print("number new size")
                #print(new_reference_vectors.number_of_vectors)
    
                new_reference_vectors.interactive_adapt_4(preferred_ranges)
                #print("number new size")
                #print(new_reference_vectors.number_of_vectors)

                #Replace the reference vectors that are not in the ROI
                #self.reference_vectors.values[solutions_remain==False] = np.vstack([reference_vectors_3.values[solutions_remain], new_reference_vectors_3.values])
                #self.reference_vectors.values_planar[solutions_remain==False] = np.vstack([reference_vectors_3.values_planar[solutions_remain], new_reference_vectors_3.values_planar])

                to_adapt = np.random.permutation(new_solutions)

                if (new_solutions>0):
                    self.reference_vectors.values[np.invert(solutions_remain),:] = new_reference_vectors.values[to_adapt,:]
                    self.reference_vectors.values_planar[np.invert(solutions_remain),:] = new_reference_vectors.values_planar[to_adapt,:]
            

            
            # Update the neighborhoods in all the cases
            distance_matrix_vectors = distance_matrix(self.reference_vectors.values_planar, self.reference_vectors.values_planar)
            # Get the closest vectors to obtain the neighborhoods
            self.neighborhoods = np.argsort(
                distance_matrix_vectors, axis=1, kind="quicksort"
            )[:, :self.n_neighbors]
            

    def check_roi_ranges(self, preferred_ranges)-> list:
        lower_limits = np.array([ranges[0] for ranges in preferred_ranges])
        upper_limits = np.array([ranges[1] for ranges in preferred_ranges])

        ub_check = (np.all(np.less_equal(self.population.fitness,upper_limits),axis=1))
        lb_check = (np.all(np.greater_equal(self.population.fitness,lower_limits),axis=1))

        keep_solutions = np.logical_and(ub_check,lb_check)
        return keep_solutions


    def _select(self, current_neighborhood, offspring_fx) -> list:
        return self.selection_operator.do(
            self.population,
            self.reference_vectors,
            self._ideal_point,
            current_neighborhood,
            offspring_fx,
        )

