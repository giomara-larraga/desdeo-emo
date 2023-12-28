from typing import Dict, Type, Tuple, Callable
import numpy as np
import pandas as pd
from desdeo_emo.population.Population import Population
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_problem import MOProblem
from desdeo_emo.EAs import BaseEA
from desdeo_tools.interaction.request import BaseRequest
from desdeo_emo.EAs.BaseEA import eaError
from desdeo_tools.interaction import (
    SimplePlotRequest,
    ReferencePointPreference,
    validate_ref_point_with_ideal_and_nadir,
)


class BaseDominanceEA(BaseEA):
    """The Base class for indicator based EAs.

    This class contains most of the code to set up the parameters and operators.
    It also contains the logic of a indicator EA.

    Parameters
    ----------
    problem : MOProblem
        The problem class object specifying the details of the problem.
    selection_operator : Type[SelectionBase], optional
        The selection operator to be used by the EA, by default None.
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
    total_function_evaluations :int, optional
        Set an upper limit to the total number of function evaluations. When set to
        zero, this argument is ignored and other termination criteria are used.
    use_surrogates: bool, optional
        A bool variable defining whether surrogate problems are to be used or
        not. By default False
    """

    def __init__(
        self,
        problem: MOProblem,
        population_size: int,  # size required
        selection_operator: Type[SelectionBase] = None,
        population_params: Dict = None,
        initial_population: Population = None,
        a_priori: bool = False,
        interact: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        use_surrogates: bool = False,
        seed: int = None,
    ):
        super().__init__(
            a_priori=a_priori,
            interact=interact,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
            selection_operator=selection_operator,
            use_surrogates=use_surrogates,
            seed=seed,
        )
        self._interaction_location = "Selection"
        if initial_population is not None:
            self.population = initial_population
        elif initial_population is None:
            self.population = Population(
                problem, population_size, population_params, use_surrogates
            )
            self._function_evaluation_count += population_size

    def end(self):
        """Conducts non-dominated sorting at the end of the evolution process
        Returns:
            tuple: The first element is a 2-D array of the decision vectors of the non-dominated solutions.
                The second element is a 2-D array of the corresponding objective values.
        """
        non_dom = self.population.non_dominated_objectives()
        return (
            self.population.individuals[non_dom, :],
            self.population.objectives[non_dom, :],
        )

    def _next_gen(self):
        """
        Run one generation of dominance based EA. Intended to be used by next_iteration.
        """
        offspring = self.population.mate()  # (params=self.params)
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

    def _select(self) -> list:
        """
        Performs the selection, returns indices of selected individuals.

        Returns
        -------
        list
            List of indices of individuals to be selected.
        """
        return self.selection_operator.do(self.population)

    def manage_preferences(self, preference=None):
        """Forward the preference to the correct preference handling method.

        Args:
            preference (_type_, optional): _description_. Defaults to None.

        Raises:
            eaError: Preference handling not implemented.
        """
        if self._interaction_location is None:
            return
        if self._interaction_location == "Population":
            try:
                self.population.manage_preferences(preference)
                return
            except NotImplementedError as e:
                raise eaError(
                    "Interaction handling not implemented in population."
                ) from e
        if self._interaction_location == "Selection":
            try:
                self.selection_operator.manage_preferences(self.population, preference)
                return
            except NotImplementedError as e:
                raise eaError(
                    "Interaction handling not implemented in selection."
                ) from e
        if self._interaction_location == "EA":
            raise eaError("Override this method in child class.")
        if self._interaction_location == "Problem":
            try:
                self.population.problem.manage_preferences(self.population, preference)
                self.population.reevaluate_fitness()
                return
            except NotImplementedError as e:
                raise eaError(
                    "Interaction handling not implemented in problem class."
                ) from e

    def request_preferences(self) -> Type[BaseRequest]:
        if self.interact is False:
            return
        if self._interaction_location == "Problem":
            return self.population.problem.request_preferences(self.population)
        if self._interaction_location == "Selection":
            return self.selection_operator.request_preferences(self.population)

    def request_plot(self) -> SimplePlotRequest:
        dimensions_data = pd.DataFrame(
            index=["minimize", "ideal", "nadir"],
            columns=self.population.problem.get_objective_names(),
        )
        dimensions_data.loc["minimize"] = self.population.problem._max_multiplier
        dimensions_data.loc["ideal"] = self.population.ideal_objective_vector
        dimensions_data.loc["nadir"] = self.population.nadir_objective_vector
        data = pd.DataFrame(
            self.population.objectives, columns=self.population.problem.objective_names
        )
        return SimplePlotRequest(
            data=data, dimensions_data=dimensions_data, message="Objective Values"
        )

    def requests(self) -> Tuple:
        return (self.request_preferences(), self.request_plot())
