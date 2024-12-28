from typing import Dict, Type, Callable
import numpy as np
from desdeo_emo.population.Population import Population
from desdeo_emo.EAs.BaseDominanceEA import BaseDominanceEA
from desdeo_tools.utilities.quality_indicator import epsilon_indicator
from desdeo_emo.selection.RNSGAII_select import RNSGAII_select
from desdeo_problem import MOProblem
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.recombination.CrossoverBase import CrossoverBase
from desdeo_emo.recombination.MutationBase import MutationBase
from desdeo_emo.selection.NTournamentSelection import NTournamentSelection
from desdeo_emo.selection.NRandomSelection import NRandomSelection
from desdeo_emo.recombination.SBX import SBX
from desdeo_emo.recombination.BLX import BLX
from desdeo_emo.recombination.BoundedPolynomialMutation import BP_mutation
from desdeo_emo.recombination.UniformMutation import UniformMutation

class RNSGAII(BaseDominanceEA):
    """Python Implementation of RNSGAII.

    Most of the relevant code is contained in the super class.
    Parameters
    ----------
    problem: MOProblem
        The problem class object specifying the details of the problem.
    population_size : int, optional
        The desired population size, by default None, which sets up a default value
        of population size depending upon the dimensionaly of the problem.
    population_params : Dict, optional
        The parameters for the population class, by default None. See
        desdeo_emo.population.Population for more details.
    initial_population : Population, optional
        An initial population class, by default None. Use this if you want to set up
        a specific starting population, such as when the output of one EA is to be
        used as the input of another.
    a_priori : bool, optional
        A bool variable defining whether a priori preference is to be used or not.
        By default False
    interact : bool, optional
        A bool variable defining whether interactive preference is to be used or
        not. By default False
    n_iterations : int, optional
        The total number of iterations to be run, by default 10. This is not a hard
        limit and is only used for an internal counter.
    n_gen_per_iter : int, optional
        The total number of generations in an iteration to be run, by default 100.
        This is not a hard limit and is only used for an internal counter.
    total_function_evaluations : int, optional
        Set an upper limit to the total number of function evaluations. When set to
        zero, this argument is ignored and other termination criteria are used.
    use_surrogates: bool, optional
        A bool variable defining whether surrogate problems are to be used or
        not. By default False
    kappa : float, optional
        Fitness scaling value for indicators. By default 0.05.
    indicator : Callable, optional
        Quality indicator to use in indicatorEAs. By default in IBEA this is additive epsilon indicator.

    """

    def __init__(
        self,
        problem: MOProblem,
        population_size: int,  # size required
        initial_population: Population = None,
        n_survive: int = None,
        selection_type: str = None,
        a_priori: bool = False,
        interact: bool = False,
        population_params: Dict = None,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        use_surrogates: bool = False,
        epsilon=0.001,
        normalization="front",
        weights=None,
        extreme_points_as_reference_points=False,
        seed: int = None, 
        crossover: str = None,
        crossover_probability: float =None,
        crossover_repair: str = None,
        crossover_distribution_index: float =None,
        blx_alpha_crossover: float = None,
        mutation: str = None,
        mutation_probability: float = None,
        mutation_repair: str = None,
        polinomial_mut_dist_index: float = None,
        uniform_mut_perturbation: float = None,
        selection_parents: str = None,
        slection_tournament_size: int = None,
    ):
        super().__init__(
            problem=problem,
            population_size=population_size,
            population_params=population_params,
            a_priori=a_priori,
            interact=interact,
            n_iterations=n_iterations,
            initial_population=initial_population,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
            use_surrogates=use_surrogates,
            seed=seed,
        )
        self.selection_type = selection_type
        self.selection_operator = RNSGAII_select(
            self.population,
            n_survive,
            selection_type=selection_type,
            epsilon=epsilon,
            normalization=normalization,
            weights=weights,
            extreme_points_as_reference_points=extreme_points_as_reference_points,
        )
        self.slection_tournament_size = slection_tournament_size
        if (selection_parents == None):
            self.selection_parents = TournamentSelection(self.population, slection_tournament_size)
        else: 
            if selection_parents == "tournament":
                self.selection_parents = NTournamentSelection(self.population, slection_tournament_size)
            elif selection_parents == "random":
                self.selection_parents = NRandomSelection(self.population, slection_tournament_size)
            else:
                print("error")
        if (crossover == None):
            self.population.xover = SBX(crossover_probability, crossover_distribution_index, crossover_repair)
        else:
            if crossover == "BLX_ALPHA":
                #print("using blx")
                self.population.xover = BLX(pop= self.population, prob=crossover_probability, alpha=blx_alpha_crossover, repair_method=crossover_repair)
            elif crossover == "SBX":
                #print("using sbx")
                self.population.xover = SBX(pop= self.population, ProC=crossover_probability, DisC=crossover_distribution_index, repair_method=crossover_repair)
            else:
                print("error")
        if (mutation == None):
            self.population.mutation = BP_mutation(self.population.problem.get_variable_lower_bounds(), self.population.problem.get_variable_upper_bounds(), ProM= mutation_probability, DisM= polinomial_mut_dist_index, repair_method= mutation_repair)
        else:
            if mutation == "uniform":
                self.population.mutation = UniformMutation(self.population.problem.get_variable_lower_bounds(), self.population.problem.get_variable_upper_bounds(), ProM= mutation_probability, PerM= uniform_mut_perturbation, repair_method= mutation_repair)
            elif mutation == "polynomial":
                self.population.mutation = BP_mutation(self.population.problem.get_variable_lower_bounds(), self.population.problem.get_variable_upper_bounds(), ProM= mutation_probability, DisM= polinomial_mut_dist_index, repair_method= mutation_repair)
            else:
                print("error")

    def _next_gen(self):
        parents = self.selection_parents.do(self.population)  
        offspring = self.population.mate(mating_individuals=parents)  # (params=self.params)
        if self.save_non_dominated:
            results = self.population.add(offspring, self.use_surrogates)
            self.non_dominated_archive(offspring, results)
        else:
            self.population.add(offspring, self.use_surrogates)
        selected = self._select()
        self.population.keep(selected)
        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        if not self.use_surrogates:
            self._function_evaluation_count += offspring.shape[0]
