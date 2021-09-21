from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII
from desdeo_emo.EAs import MOEA_D
from desdeo_emo.utilities.plotlyanimate import animate_init_, animate_next_


dtlz3 = test_problem_builder("DTLZ2", n_of_variables=12, n_of_objectives=3)
evolver = MOEA_D(dtlz3, n_iterations=10)
figure = animate_init_(evolver.population.objectives, filename="dtlz3.html")
while evolver.continue_evolution():
    evolver.iterate()
    figure = animate_next_(
        evolver.population.objectives,
        figure,
        filename="dtlz3.html",
        generation=evolver._iteration_counter,
    )
