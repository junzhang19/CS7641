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

OUTPUT_FILE = 'tsm_result.csv'

SA_TEMPERATURE = 1e12
SA_COOLING_FACTOR = .999

GA_POPULATION = 2000
GA_CROSSOVER = 1500
GA_MUTATION = 250

MIMIC_SAMPLES = 500
MIMIC_TO_KEEP = 100

# Random number generator */
N=50
random = Random()

cycle = 5
iterations = [10,20,30,50,70,100,200,300,500,700,1000,2000,3000,4000,5000]

rhc_fitness = [[] for i in range(cycle)]
rhc_training_time = [[] for i in range(cycle)]

sa_fitness = [[] for i in range(cycle)]
sa_training_time = [[] for i in range(cycle)]

ga_fitness = [[] for i in range(cycle)]
ga_training_time = [[] for i in range(cycle)]

mimic_fitness = [[] for i in range(cycle)]
mimic_training_time = [[] for i in range(cycle)]

for n in range(cycle):
    print("the %d th cycle" %(n+1))
    
    points = [[0 for x in xrange(2)] for x in xrange(N)]
    for i in range(0, len(points)):
        points[i][0] = random.nextDouble()
        points[i][1] = random.nextDouble()
    
    ef = TravelingSalesmanRouteEvaluationFunction(points)
    odd = DiscretePermutationDistribution(N)
    nf = SwapNeighbor()
    mf = SwapMutation()
    cf = TravelingSalesmanCrossOver(ef)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
    
    ef2 = TravelingSalesmanRouteEvaluationFunction(points)
    fill = [N] * N
    ranges = array('i', fill)
    odd2 = DiscreteUniformDistribution(ranges);
    df = DiscreteDependencyTree(.1, ranges); 
    pop = GenericProbabilisticOptimizationProblem(ef2, odd2, df);
    
    rhc = RandomizedHillClimbing(hcp)
    sa = SimulatedAnnealing(SA_TEMPERATURE, SA_COOLING_FACTOR, hcp)
    ga = StandardGeneticAlgorithm(GA_POPULATION, GA_CROSSOVER, GA_MUTATION, gap)
    mimic = MIMIC(MIMIC_SAMPLES, MIMIC_TO_KEEP, pop)
    
    for n_iteration in iterations:
        fit_rhc = FixedIterationTrainer(rhc, n_iteration*200)
        fit_sa = FixedIterationTrainer(sa, n_iteration*200)
        fit_ga = FixedIterationTrainer(ga, n_iteration)
        fit_mimic = FixedIterationTrainer(mimic, n_iteration)
        
        print("calculating the %d th iteration" % n_iteration)

        # Training
        start_rhc = time.time()
        fit_rhc.train()
        end_rhc = time.time()

        start_sa = time.time()
        fit_sa.train()
        end_sa = time.time()
    
        start_ga = time.time()
        fit_ga.train()
        end_ga = time.time()
    
        start_mimic = time.time()
        fit_mimic.train()
        end_mimic = time.time()

        # Result extracting
        last_training_time_rhc = end_rhc - start_rhc
        rhc_training_time[n].append(last_training_time_rhc)
        rhc_fitness[n].append(ef.value(rhc.getOptimal()))

        last_training_time_sa = end_sa - start_sa
        sa_training_time[n].append(last_training_time_sa)
        sa_fitness[n].append(ef.value(sa.getOptimal()))
        
        last_training_time_ga = end_ga - start_ga
        ga_training_time[n].append(last_training_time_ga)
        ga_fitness[n].append(ef.value(ga.getOptimal()))

        last_training_time_mimic = end_mimic - start_mimic
        mimic_training_time[n].append(last_training_time_mimic)
        mimic_fitness[n].append(ef.value(mimic.getOptimal()))

overall_rhc_training_time = list_avg(*rhc_training_time)
overall_rhc_fitness = list_avg(*rhc_fitness)

overall_sa_training_time = list_avg(*sa_training_time)
overall_sa_fitness = list_avg(*sa_fitness)

overall_ga_training_time = list_avg(*ga_training_time)
overall_ga_fitness = list_avg(*ga_fitness)

overall_mimic_training_time = list_avg(*mimic_training_time)
overall_mimic_fitness = list_avg(*mimic_fitness)

#print mimic_training_time
#print overall_mimic_training_time

with open(OUTPUT_FILE, "w") as outFile:
    for i in range(1):
        outFile.write(','.join([
            "sa_rhc_iterations",
            "rhc_fitness",
            "rhc_training_time",
            "sa_fitness",
            "sa_training_time",
            "ga_mimic_iterations"
            "ga_fitness",
            "ga_training_time",
            "mimic_fitness",
            "mimic_training_time"]) + '\n')
    for i in range(len(iterations)):
        outFile.write(','.join([
            str(iterations[i]*200),
            str(overall_rhc_fitness[i]),
            str(overall_rhc_training_time[i]),
            str(overall_sa_fitness[i]),
            str(overall_sa_training_time[i]),
            str(iterations[i]),
            str(overall_ga_fitness[i]),
            str(overall_ga_training_time[i]),
            str(overall_mimic_fitness[i]),
            str(overall_mimic_training_time[i])]) + '\n')
        
print("The end of the program") 
            
            