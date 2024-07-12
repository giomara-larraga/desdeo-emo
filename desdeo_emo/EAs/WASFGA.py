from typing import Dict

import numpy as np
from desdeo_emo.EAs.BaseEA import BaseDecompositionEA
from desdeo_emo.population.Population import Population
from desdeo_problem import MOProblem

from desdeo_emo.selection.WASFGA_select import WASFGA_select


class WASF_GA(BaseDecompositionEA):
    """Python implementation of WASF-GA

    .. Ruiz, A.B., Saborido, R. & Luque, M. A preference-based evolutionary algorithm for multiobjective optimization: 
    the weighting achievement scalarizing function genetic algorithm. J Glob Optim 62, 101–129 (2015). 
    https://doi.org/10.1007/s10898-014-0214-y

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
        reference_point: np.array = None,
        population_size: int = None,
        population_params: Dict = None,
        initial_population: Population = None,
        lattice_resolution: int = None,
        interact: bool = False,
        use_surrogates: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        keep_archive: bool = False,
    ):
        super().__init__(
            problem=problem,
            population_size=population_size,
            population_params=population_params,
            initial_population=initial_population,
            lattice_resolution=lattice_resolution,
            interact=interact,
            use_surrogates=use_surrogates,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
            keep_archive=keep_archive,
        )
        #self.population_size = self.reference_vectors.number_of_vectors
        self.problem = problem
        self.reference_point = reference_point

        selection_operator = WASFGA_select(
            self.population
        )
        self.selection_operator = selection_operator


    def _select(self) -> list:
        return self.selection_operator.do(
            self.population, self.reference_point
        )
