from __future__ import with_statement
import sys
import os
import time

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
from array import array

def list_avg(*args):
    output = []
    for i in range(len(args[0])):
        temp = 0.0
        for j in range(len(args)):
            temp += args[j][i]
        output.append(temp / len(args))
    return output

OUTPUT_FILE = 'ks_result_MIMIC.csv'

MIMIC_SAMPLES = 200
MIMIC_TO_KEEP_pool = [5,10,20,50,100]

# Random number generator */
random = Random()
# The number of items
NUM_ITEMS = 40
# The number of copies each
COPIES_EACH = 4
# The maximum weight for a single element
MAX_WEIGHT = 50
# The maximum volume for a single element
MAX_VOLUME = 50
# The volume of the knapsack 
KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

fill = [COPIES_EACH] * NUM_ITEMS
copies = array('i', fill)

cycle = 5
n_iteration = 10

mimic_fitness = [[] for i in range(cycle)]
mimic_training_time = [[] for i in range(cycle)]

for n in range(cycle):
    print("the %d th cycle" %(n+1))
    fill = [0] * NUM_ITEMS
    weights = array('d', fill)
    volumes = array('d', fill)
    for i in range(0, NUM_ITEMS):
		weights[i] = random.nextDouble() * MAX_WEIGHT
		volumes[i] = random.nextDouble() * MAX_VOLUME
	
    fill = [COPIES_EACH + 1] * NUM_ITEMS
    ranges = array('i', fill)
    
    ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
    odd = DiscreteUniformDistribution(ranges)
    df = DiscreteDependencyTree(.1, ranges)
    pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
	
    for MIMIC_TO_KEEP in MIMIC_TO_KEEP_pool:
	    mimic = MIMIC(MIMIC_SAMPLES, MIMIC_TO_KEEP, pop)
	    fit_mimic = FixedIterationTrainer(mimic, n_iteration)
	    
	    print("calculating for MIMIC_TO_KEEP = %d" % MIMIC_TO_KEEP)

	    # Training
	    start_mimic = time.time()
	    fit_mimic.train()
	    end_mimic = time.time()

	    # Result extracting
	    last_training_time_mimic = end_mimic - start_mimic
	    mimic_training_time[n].append(last_training_time_mimic)
	    mimic_fitness[n].append(ef.value(mimic.getOptimal()))

overall_mimic_training_time = list_avg(*mimic_training_time)
overall_mimic_fitness = list_avg(*mimic_fitness)

with open(OUTPUT_FILE, "w") as outFile:
	for i in range(1):
		outFile.write(','.join([
            "MIMIC_TO_KEEP",
            "overall_mimic_fitness",
            "overall_mimic_training_time"]) + '\n')
	for i in range(len(MIMIC_TO_KEEP_pool)):
		outFile.write(','.join([
            str(MIMIC_TO_KEEP_pool[i]),
            str(overall_mimic_fitness[i]),
            str(overall_mimic_training_time[i])]) + '\n')
            
            