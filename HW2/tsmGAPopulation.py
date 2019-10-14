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
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
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
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays

from array import array

def list_avg(*args):
    output = []
    for i in range(len(args[0])):
        temp = 0.0
        for j in range(len(args)):
            temp += args[j][i]
        output.append(temp / len(args))
    return output

OUTPUT_FILE = 'tsm_result_GA_population.csv'

GA_POPULATION_pool = [100,200,500,1000,2000,5000,10000,20000,50000]
GA_CROSSOVER_ratio = 0.75
GA_MUTATION = 250

# Random number generator */
N=50
random = Random()

cycle = 1
n_iteration = 1000

ga_fitness = [[] for i in range(cycle)]
ga_training_time = [[] for i in range(cycle)]

for n in range(cycle):
    print("the %d th cycle" %(n+1))
    
    points = [[0 for x in xrange(2)] for x in xrange(N)]
    for i in range(0, len(points)):
        points[i][0] = random.nextDouble()
        points[i][1] = random.nextDouble()
    
    ef = TravelingSalesmanRouteEvaluationFunction(points)
    odd = DiscretePermutationDistribution(N)
    mf = SwapMutation()
    cf = TravelingSalesmanCrossOver(ef)
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
            
    for GA_POPULATION in GA_POPULATION_pool:
        GA_CROSSOVER = int(GA_POPULATION * GA_CROSSOVER_ratio)
        ga = StandardGeneticAlgorithm(GA_POPULATION, GA_CROSSOVER, GA_MUTATION, gap)
        fit_ga = FixedIterationTrainer(ga, n_iteration)
        
        print("calculating for population = %d " % GA_POPULATION)

        # Training
        start_ga = time.time()
        fit_ga.train()
        end_ga = time.time()
    
        # Result extracting
        last_training_time_ga = end_ga - start_ga
        ga_training_time[n].append(last_training_time_ga)
        ga_fitness[n].append(ef.value(ga.getOptimal()))

overall_ga_training_time = list_avg(*ga_training_time)
overall_ga_fitness = list_avg(*ga_fitness)

with open(OUTPUT_FILE, "w") as outFile:
    for i in range(1):
        outFile.write(','.join([
            "ga_populations",
            "ga_fitness",
            "ga_training_time"]) + '\n')
    for i in range(len(GA_POPULATION_pool)):
        outFile.write(','.join([
            str(GA_POPULATION_pool[i]),
            str(overall_ga_fitness[i]),
            str(overall_ga_training_time[i])]) + '\n')
        
print("The end of the program") 
            
            