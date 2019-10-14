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
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
from array import array

def list_avg(*args):
    output = []
    for i in range(len(args[0])):
        temp = 0.0
        for j in range(len(args)):
            temp += args[j][i]
        output.append(temp / len(args))
    return output

OUTPUT_FILE = 'cp_result_SA_temperature.csv'

SA_TEMPERATURE_pool = [1e1,1e3,1e5,1e7,1e9,1e11,1e13,1e15,1e17,1e19]
SA_COOLING_FACTOR = 0.95

# Random number generator */
N=60
T=N/10
fill = [2] * N
ranges = array('i', fill)

cycle = 1
n_iteration = 5000

sa_fitness = [[] for i in range(cycle)]
sa_training_time = [[] for i in range(cycle)]

for n in range(cycle):
    print("the %d th cycle" %(n+1))
    
    ef = ContinuousPeaksEvaluationFunction(T)
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    
    for SA_TEMPERATURE in SA_TEMPERATURE_pool:
        sa = SimulatedAnnealing(SA_TEMPERATURE, SA_COOLING_FACTOR, hcp)
        fit_sa = FixedIterationTrainer(sa, n_iteration)
        
        print("calculating for temperature = %e" % SA_TEMPERATURE)

        # Training
        start_sa = time.time()
        fit_sa.train()
        end_sa = time.time()
    
        # Result extracting
        last_training_time_sa = end_sa - start_sa
        sa_training_time[n].append(last_training_time_sa)
        sa_fitness[n].append(ef.value(sa.getOptimal()))
        
overall_sa_training_time = list_avg(*sa_training_time)
overall_sa_fitness = list_avg(*sa_fitness)

with open(OUTPUT_FILE, "w") as outFile:
    for i in range(1):
        outFile.write(','.join([
            "sa_temperature",
            "sa_fitness",
            "sa_training_time"]) + '\n')
    for i in range(len(SA_TEMPERATURE_pool)):
        outFile.write(','.join([
            str(SA_TEMPERATURE_pool[i]),
            str(overall_sa_fitness[i]),
            str(overall_sa_training_time[i])]) + '\n')
        
print("The end of the program") 
            
            