from desdeo_emo.EAs.IKRVEA import IK_RVEA
from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from desdeo_problem import ExperimentalProblem
from visualizations import plot_parallel_coordinates
import sys

sys.path.append("NumPy_path")
import numpy as np
import pandas as pd

from sklearn.gaussian_process.kernels import Matern
from pymoo.problems import get_problem
import copy
from desdeo_tools.scalarization.ASF import SimpleASF


def select_solutions(objectives, problem, pref, u):
    asf_values = SimpleASF([1] * problem.n_of_objectives).__call__(
        objectives, pref.response.values
    )
    idx = np.argpartition(asf_values, u)[:u]  # indices of best solutions based on ASF
    Best_solutions = objectives[idx]
    return Best_solutions


def obj_function1(x):

    out = {
        "F": "",
        "G": "",
    }
    problem = get_problem("dtlz2", 12)
    problem._evaluate(x, out)
    return out["F"][:, 0]


def obj_function2(x):

    out = {
        "F": "",
        "G": "",
    }
    problem = get_problem("dtlz2", 12)
    problem._evaluate(x, out)
    return out["F"][:, 1]


def obj_function3(x):

    out = {
        "F": "",
        "G": "",
    }
    problem = get_problem("dtlz2", 12)
    problem._evaluate(x, out)
    return out["F"][:, 2]


if __name__ == "__main__":
    refpoint = np.asarray([0.2, 0.5, 0.9])
    n_obj = 3
    n_var = n_obj + 9
    var_names = ["x" + str(i + 1) for i in range(n_var)]
    obj_names = ["f" + str(i + 1) for i in range(n_obj)]
    unc_names = ["unc" + str(i + 1) for i in range(n_obj)]

    # 1) Generate initial population
    x = np.random.random((120, n_var))
    initial_obj = {
        "F": "",
        "G": "",
    }
    get_problem("dtlz2", 12)._evaluate(x, initial_obj)

    data = np.hstack((x, initial_obj["F"]))
    datapd = pd.DataFrame(data=data, columns=var_names + obj_names)

    problem = ExperimentalProblem(
        data=datapd,
        objective_names=obj_names,
        variable_names=var_names,
        uncertainity_names=unc_names,
        evaluators=[obj_function1, obj_function2, obj_function3],
    )

    # 2) Train Kriging models for each expensive objective
    problem.train(
        models=GaussianProcessRegressor, model_parameters={"kernel": Matern(nu=1.5)}
    )
    u = 10  # number of solutions that we use to update surrogates in each iteration

    # 3) Run an iteration of Interactive KRVEA
    evolver = IK_RVEA(
        problem,
        interact=True,
        n_iterations=1,
        n_gen_per_iter=100,
        lattice_resolution=10,
        use_surrogates=True,
        population_size=120,
        number_of_update=u,
    )
    evolver.set_interaction_type("Reference point")

    i = 0
    while i <= 5:
        pref, plot2 = copy.deepcopy(evolver.requests())
        pref.response = pd.DataFrame(
            [refpoint], columns=pref.content["dimensions_data"].columns
        )
        plot = evolver.iterate(pref)
        i += 1

    objectives = problem.archive.drop(problem.variable_names, axis=1).to_numpy()

    Best_solutions = select_solutions(objectives, problem, pref, u)

    plot_parallel_coordinates(Best_solutions, reference_point=refpoint)
