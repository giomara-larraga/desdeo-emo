
from typing import Dict, Union, List

from desdeo_emo.EAs.BaseEA import eaError
from desdeo_emo.EAs.BaseEA import BaseDecompositionEA, BaseEA
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.population.Population import Population
from desdeo_emo.selection.IOPIS_APD import IOPIS_APD_Select
from desdeo_emo.selection.IOPIS_NSGAIII import IOPIS_NSGAIII_select
from desdeo_problem import MOProblem
from desdeo_tools.scalarization import StomASF, PointMethodASF, AugmentedGuessASF
from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors
from desdeo_tools.interaction import (
    ReferencePointPreference,
    validate_ref_point_with_ideal_and_nadir,
)
import numpy as np
import pandas as pd

class NUMS():
    def __init__(
        self,
        problem: MOProblem,
        lattice_resolution: int = None,
        roi_size: float = 0,
        reference_point = None,

    ):
        num_obj = problem.n_of_objectives
        self.roi_size = roi_size
        self.reference_point = reference_point

        # 1) Initialize reference vectors
        if lattice_resolution is None:
            lattice_res_options = [49, 13, 7, 5, 4, 3, 3, 3, 3]
            if num_obj < 11:
                lattice_resolution = lattice_res_options[num_obj - 2]
            else:
                lattice_resolution = 3
        self.reference_vectors = ReferenceVectors(lattice_resolution, num_obj)

        

       
    def find_pivot_point(self):
        # 2 ) Find the pivot point
        self.reference_point

    def manage_preferences(self, preference=None):
        """Run the interruption phase of EA.

        Use this phase to make changes to RVEA.params or other objects.
        Updates Reference Vectors (adaptation), conducts interaction with the user.
        """
        if preference is None:
            msg = "Giving preferences is mandatory"
            raise eaError(msg)

        if not isinstance(preference, ReferencePointPreference):
            msg = (
                f"Wrong object sent as preference. Expected type = "
                f"{type(ReferencePointPreference)} or None\n"
                f"Recieved type = {type(preference)}"
            )
            raise eaError(msg)

        if preference.request_id != self._interaction_request_id:
            msg = (
                f"Wrong preference object sent. Expected id = "
                f"{self._interaction_request_id}.\n"
                f"Recieved id = {preference.request_id}"
            )
            raise eaError(msg)

        refpoint = preference.response.values * self.population.problem._max_multiplier
        self._preference = refpoint
        scalarized_space_fitness = np.asarray(
            [
                scalar(self.population.fitness, self._preference)
                for scalar in self.scalarization_methods
            ]
        ).T
        self.reference_vectors.adapt(scalarized_space_fitness)
        self.reference_vectors.neighbouring_angles()

    def request_preferences(self) -> Union[None, ReferencePointPreference]:
        dimensions_data = pd.DataFrame(
            index=["minimize", "ideal", "nadir"],
            columns=self.population.problem.get_objective_names(),
        )
        dimensions_data.loc["minimize"] = self.population.problem._max_multiplier
        dimensions_data.loc["ideal"] = self.population.ideal_objective_vector
        dimensions_data.loc["nadir"] = self.population.nadir_objective_vector
        message = (
            f"Provide a reference point worse than to the ideal point and better than"
            f" the nadir point.\n"
            f"Ideal point: \n{dimensions_data.loc['ideal']}\n"
            f"Nadir point: \n{dimensions_data.loc['nadir']}\n"
            f"The reference point will be used to create scalarization functions in "
            f"the preferred region.\n"
        )
        interaction_priority = "required"
        self._interaction_request_id = np.random.randint(0, 1e7)
        return ReferencePointPreference(
            dimensions_data=dimensions_data,
            message=message,
            interaction_priority=interaction_priority,
            preference_validator=validate_ref_point_with_ideal_and_nadir,
            request_id=self._interaction_request_id,
        )

    def _select(self) -> List:
        return self.selection_operator.do(
            self.population, self.reference_vectors, self._preference
        )
