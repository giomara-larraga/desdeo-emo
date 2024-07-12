import numba
import numpy as np
from desdeo_emo.population import Population
from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors
from desdeo_tools.scalarization.ASF import ASFBase
from desdeo_tools.utilities import fast_non_dominated_sort

class WASFGARanking:
    """Class object for reference vectors."""

    def __init__(
        self,
        utilityFunction: ASFBase,
    ):
        self.numberOfRanks = 0
        self.numberOfWeights = 0
        self.utilityFunction = utilityFunction
        self.rankedSubpopulations = []
        self.sf_values = np.array([])
        #self.numberOfViolatedConstraints = new NumberOfViolatedConstraints<S>() ;
        #self.overallConstraintViolation = new OverallConstraintViolation<S>();

    def _compute_feasible(self, data: np.ndarray, referenceVectors: ReferenceVectors, reference_point: np.array):
        num_solutions = len(data)
        num_vectors = referenceVectors.number_of_vectors
        self.sf_values = np.zeros((num_solutions, num_vectors))
        #Compute SF for all solutions
        for i in range(0, num_solutions):
            for j in range(0, num_vectors):
                self.sf_values[i][j] = self.utilityFunction.__call__(data[i], reference_point, referenceVectors.values[j])
        
        print(self.sf_values)
        return fast_non_dominated_sort(self.sf_values)

    
    def _compute(self, population:Population, referenceVectors: ReferenceVectors, reference_point: np.array):
        """Create the reference vectors.

        Parameters
        ----------
        creation_type : str, optional
            'Uniform' creates the reference vectors uniformly using simplex lattice
            design. 'Focused' creates reference vectors symmetrically around a central
            reference vector. By default 'Uniform'.
        """

        self.popularion_ = population

        numberOfWeights = referenceVectors.number_of_vectors
        violation_values = population.constraint
        #Split the population in feasible and unfeasible solutions
        feasibleSolutions = population
        unfeasibleSolutions = population


        if population.problem.n_of_constraints>0:
            violation_values = np.maximum(0, violation_values)
            feasible_bool = (violation_values == 0).all(axis=1)
            feasibleSolutions = feasibleSolutions.keep(np.where(feasible_bool))
            unfeasibleSolutions = unfeasibleSolutions.keep(np.where(feasible_bool==False))
            feasibleSize = len(np.where(feasible_bool))
            unfeasibleSize = len(np.where(feasible_bool==False))
            violation_values = violation_values[feasible_bool==False]
        else:
            #violation_values=np.zeros(population.pop_size)
            feasibleSolutions = population
            unfeasibleSolutions = []
            feasibleSize = population.pop_size
            unfeasibleSize =0

        #Compute the number of fronts for feasible solutions
        if(feasibleSize > 0):
            if(feasibleSize > numberOfWeights):
                numberOfRanksForFeasibleSolutions = int((feasibleSize + 1) / numberOfWeights)
            else:
                numberOfRanksForFeasibleSolutions = 1
        else:
            numberOfRanksForFeasibleSolutions = 0

        #Each unfeasible solution goes to a different front
        numberOfRanksForUnfeasibleSolutions = unfeasibleSize

        #Initialization of properties
        self.numberOfRanks = numberOfRanksForFeasibleSolutions + numberOfRanksForUnfeasibleSolutions
        self.rankedSubpopulations =  [[] for _ in range(self.numberOfRanks)]
        
		#Classification of feasible solutions
        if(feasibleSize>0):
            #Iteration for each front
            for index in range(0, numberOfRanksForFeasibleSolutions):
                #Iteration for each weight vector
                for indexOfWeight in range(0, numberOfWeights):
                    if(feasibleSize>0):
                        indexOfBestSolution = 0
                        minimumValue = self.utilityFunction.__call__(feasibleSolutions.objectives[0], reference_point, referenceVectors.values[indexOfWeight])
                        for solutionIdx in range(1,feasibleSize):
                            value = self.utilityFunction.__call__(feasibleSolutions.objectives[solutionIdx], reference_point, referenceVectors.values[indexOfWeight])
                            if (value < minimumValue):
                                minimumValue = value
                                indexOfBestSolution = solutionIdx
                        #Introduce the best feasible individual for the current weight vector into the current front
                        self.rankedSubpopulations[index].append(feasibleSolutions.objectives[indexOfBestSolution])
                        feasibleSolutions.delete([indexOfBestSolution])
                        feasibleSize = feasibleSize - 1
                        #solutionToInsert = feasibleSolutions.remove(indexOfBestSolution)
                
        if (unfeasibleSize>0):
			#Obtain the rank of each unfeasible solution
            rankForUnfeasibleSolutions = self.rankUnfeasibleSolutions(unfeasibleSolutions, unfeasibleSize,violation_values,reference_point, referenceVectors)

			#Add each unfeasible solution into their corresponding front
            for index in range(len(rankForUnfeasibleSolutions)):
                solutionToInsert = unfeasibleSolutions[index]
                rank = rankForUnfeasibleSolutions[index] + numberOfRanksForFeasibleSolutions
                #setAttribute(solutionToInsert, rank);
                self.rankedSubpopulations[rank].append(solutionToInsert)

        return self.rankedSubpopulations
		  

    def getSubfront(self, rank:int):
        return self.rankedSubpopulations[rank]

    def getNumberOfSubFronts(self):
        return len(self.rankedSubpopulations)
	
    def rankUnfeasibleSolutions(self, population:Population, popsize:int, violationValues:list, reference_point: np.ndarray, referenceVectors: ReferenceVectors):
        rank = np.zeros(popsize)
		
		#Iteration for each solution
        for indexOfFirstSolution in range(0, popsize-1):
            #The current solution is compared with the following ones
            for indexOfSecondSolution in range(indexOfFirstSolution + 1, popsize):
                numberOfViolatedConstraintsBySolution1 = violationValues[indexOfFirstSolution]
                numberOfViolatedConstraintsBySolution2 = violationValues[indexOfSecondSolution]
                #//The number of violated constraints is compared.
                #//A solution with higher number of violated constraints has a worse (higher) rank
                if (numberOfViolatedConstraintsBySolution1 > numberOfViolatedConstraintsBySolution2):
                    rank[indexOfFirstSolution] = rank[indexOfFirstSolution] +1
                elif (numberOfViolatedConstraintsBySolution1 < numberOfViolatedConstraintsBySolution2):
                    rank[indexOfSecondSolution] = rank[indexOfSecondSolution] +1
                else:
					#Because the solutions have a similar violated number of constraints, the overall constraint
					#violation values are compared.
					#Note that overall constraint values are negative in jMetal
                    overallConstraintViolationSolution1 = np.sum(violationValues[indexOfFirstSolution])
                    overallConstraintViolationSolution2 = np.sum(violationValues[indexOfSecondSolution])

					#The overall constraint violation values are compared.
					#Note that overall constraint values are negative in jMetal.
					#Thus, a solution with higher value has a better (higher) rank
                    if (overallConstraintViolationSolution1 < overallConstraintViolationSolution2):
                        rank[indexOfSecondSolution] = rank[indexOfSecondSolution] + 1
                    elif (overallConstraintViolationSolution1 > overallConstraintViolationSolution2):
                        rank[indexOfFirstSolution] = rank[indexOfFirstSolution] + 1
                    else:
                        #Because the solutions have the same overall constraint violation values, we compare the
                        #the value of their utility functions. Lower values are better.
                        minimumValueFirstSolution = minimumValueSecondSolution = float("inf")
                    for indexOfWeight in range(0, self.numberOfWeights):
                        value = self.utilityFunction.__call__(population.objectives[indexOfFirstSolution],reference_point, referenceVectors.values[indexOfWeight])
                        if (value < minimumValueFirstSolution):
                            minimumValueFirstSolution = value

                        value = self.utilityFunction(population.objectives[indexOfSecondSolution],reference_point, referenceVectors.values[indexOfWeight])
                        if (value < minimumValueSecondSolution):
                            minimumValueSecondSolution = value
                    	

                    if (minimumValueFirstSolution < minimumValueSecondSolution):
                        rank[indexOfSecondSolution] = rank[indexOfSecondSolution] + 1
                    else:
                        rank[indexOfFirstSolution] = rank[indexOfFirstSolution] + 1
        return rank