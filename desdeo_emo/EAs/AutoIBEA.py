from typing import Callable, Dict, Union

from desdeo_emo.EAs.BaseEA import eaError
from desdeo_emo.EAs.BaseIndicatorEA import BaseIndicatorEA
from desdeo_emo.population.Population import Population
from desdeo_emo.selection.NSGAIII_select import NSGAIII_select
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
from desdeo_tools.utilities.quality_indicator import epsilon_indicator
from desdeo_emo.selection.TournamentSelection import TournamentSelection
import numpy as np


class AutoIBEA(BaseIndicatorEA):
    """Python Implementation of NSGA-III. Based on the pymoo package.

    Most of the relevant code is contained in the super class. This class just assigns
    the NSGAIII selection operator to BaseDecompositionEA.

    Parameters
    ----------
    problem : MOProblem
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
    lattice_resolution : int, optional
        The number of divisions along individual axes in the objective space to be
        used while creating the reference vector lattice by the simplex lattice
        design. By default None
    selection_type : str, optional
        One of ["mean", "optimistic", "robust"]. To be used in data-driven optimization.
        To be used only with surrogate models which return an "uncertainity" factor.
        Using "mean" is equivalent to using the mean predicted values from the surrogate
        models and is the default case.
        Using "optimistic" results in using (mean - uncertainity) values from the
        the surrogate models as the predicted value (in case of minimization). It is
        (mean + uncertainity for maximization).
        Using "robust" is the opposite of using "optimistic".
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
    total_function_evaluations :int, optional
        Set an upper limit to the total number of function evaluations. When set to
        zero, this argument is ignored and other termination criteria are used.
    """

    def __init__(
        self,
        problem: MOProblem,
        population_size: int, # size required
        initial_population: Population = None,
        a_priori: bool = False,
        interact: bool = False,
        population_params: Dict = None,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        use_surrogates: bool = False,
        kappa: float = 0.05, # fitness scaling ratio
        indicator: Callable = epsilon_indicator, # default indicator is epsilon_indicator
        seed: int = None,
        crossover: str = "SBX",
        crossover_probability: float =None,
        crossover_repair: str = "round",
        crossover_distribution_index: float =None,
        blx_alpha_crossover: float = None,
        mutation: str = "polynomial",
        mutation_probability: float = None,
        mutation_repair: str = "round",
        polinomial_mut_dist_index: float = None,
        uniform_mut_perturbation: float = None,
        selection_parents: str = "random",
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
            fitnesses = None,
            seed = seed,
        )
        
        self.indicator = indicator
        self.kappa = kappa
        # self.allowable_interaction_types = "Reference Point"
        # self.set_interaction_type = "Selection"
       
        selection_operator = TournamentSelection(self.population, 2)
        self.selection_operator = selection_operator



        if selection_parents == "tournament":
            self.selection_parents = NTournamentSelection(self.population, slection_tournament_size)
        elif selection_parents == "random":
            self.selection_parents = NRandomSelection(self.population, slection_tournament_size)
        else:
            print("error")

        if crossover == "BLX_ALPHA":
            #print("using blx")
            self.population.xover = BLX(pop= self.population, prob=crossover_probability, alpha=blx_alpha_crossover, repair_method=crossover_repair)
        elif crossover == "SBX":
            #print("using sbx")
            self.population.xover = SBX(pop= self.population, ProC=crossover_probability, DisC=crossover_distribution_index, repair_method=crossover_repair)
        else:
            print("error")
    

        if mutation == "uniform":
            self.population.mutation = UniformMutation(self.population.problem.get_variable_lower_bounds(), self.population.problem.get_variable_upper_bounds(), ProM= mutation_probability, PerM= uniform_mut_perturbation, repair_method= mutation_repair)
        elif mutation == "polynomial":
            self.population.mutation = BP_mutation(self.population.problem.get_variable_lower_bounds(), self.population.problem.get_variable_upper_bounds(), ProM= mutation_probability, DisM= polinomial_mut_dist_index, repair_method= mutation_repair)
        else:
            print("error")

    def _next_gen(self):
        """Run one generation of decomposition based EA. Intended to be used by
        next_iteration.
        """
        # calls fitness assignment
        self.fitnesses = np.zeros_like(self.population.fitness)
        self.fitnesses = self._fitness_assignment(self.fitnesses)
        # performs the enviromental selection
        while (self.population.pop_size < self.population.individuals.shape[0]):
            worst_index = np.argmin(self.fitnesses, axis=0)[0] # gets the index worst member of population
            self.fitnesses = self._environmental_selection(self.fitnesses, worst_index)
            # remove the worst member from population and from fitnesses
            self.population.delete(worst_index)
            self.fitnesses = np.delete(self.fitnesses, worst_index, 0)

        # perform binary tournament selection. 
        chosen = self._select()
        offspring = self.population.mate(mating_individuals=chosen)  # (params=self.params)
        if self.save_non_dominated:
            results = self.population.add(offspring, self.use_surrogates)
            self.non_dominated_archive(offspring, results)
        else:
            self.population.add(offspring, self.use_surrogates)
        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        if not self.use_surrogates:
            self._function_evaluation_count += offspring.shape[0]

    def _fitness_assignment(self, fitnesses):
        """
            Performs the fitness assignment of the individuals.
        """
        for i in range(self.population.individuals.shape[0]):
            for j in range(self.population.individuals.shape[0]):
                if j != i:
                   fitnesses[i] += -np.exp(-self.indicator(self.population.fitness[i], 
                        self.population.fitness[j]) / self.kappa)
        return fitnesses


    def _environmental_selection(self, fitnesses, worst_index):
        """
            Selects the worst member of population, then updates the population members fitness values compared to the worst individual.
            Worst individual is removed from the population.
        """
        # updates the fitness values
        for i in range(self.population.individuals.shape[0]):
            if worst_index != i:
                fitnesses[i] += np.exp(-self.indicator(self.population.fitness[i], self.population.fitness[worst_index]) / self.kappa)
        return fitnesses