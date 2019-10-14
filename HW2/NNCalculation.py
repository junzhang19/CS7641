"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network that is classifying abalone as having either fewer
or more than 15 rings.

Based on AbaloneTest.java by Hannah Lau
"""
from __future__ import with_statement

import os
import csv
import time

from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm

# INPUT_FILE = os.path.join("..", "src", "opt", "test", "abalone.txt")
INPUT_FILE = "african_crises.csv"
OUTPUT_FILE = 'nn_result.csv'

INPUT_LAYER = 9
HIDDEN_LAYER = 9
OUTPUT_LAYER = 1
TRAINING_ITERATIONS_pool = [10]#,20,50,100,200,500,1000,2000,5000] #1000


def initialize_instances():
    """Read the data into a list of instances."""
    instances = []

    # Read in the CSV file
    with open(INPUT_FILE, "r") as abalone:
        reader = csv.reader(abalone)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(1 if row[-1] == "crisis" else 0))
            instances.append(instance)

    return instances


def train(oa, network, oaName, instances, measure,TRAINING_ITERATIONS):
    """Train a given network on a set of instances.

    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] instances:
    :param AbstractErrorMeasure measure:
    """
    # print "\nError results for %s\n---------------------------" % (oaName,)

    for iteration in xrange(TRAINING_ITERATIONS):
        oa.train()
        '''
        error = 0.00
        for instance in instances:
            network.setInputValues(instance.getData())
            network.run()

            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)

        # print "%0.03f" % error
        '''

def main():
    
    accuracies = [[] for i in range(3)]
    training_times = [[] for i in range(3)]

    """Run algorithms on the dataset."""
    instances = initialize_instances()
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(instances)

    for TRAINING_ITERATIONS in TRAINING_ITERATIONS_pool:
        print("Calculating with %d iterations" % TRAINING_ITERATIONS)
        networks = []  # BackPropagationNetwork
        nnop = []  # NeuralNetworkOptimizationProblem
        oa = []  # OptimizationAlgorithm
        oa_names = ["RHC", "SA", "GA"]
        results = ""

        for name in oa_names:
            classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
            networks.append(classification_network)
            nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))
            
        oa.append(RandomizedHillClimbing(nnop[0]))
        oa.append(SimulatedAnnealing(1E11, .95, nnop[1]))
        oa.append(StandardGeneticAlgorithm(200, 100, 10, nnop[2]))
    
        for i, name in enumerate(oa_names):
            start = time.time()
            correct = 0
            incorrect = 0

            train(oa[i], networks[i], oa_names[i], instances, measure,TRAINING_ITERATIONS)
            end = time.time()
            training_time = end - start

            optimal_instance = oa[i].getOptimal()
            networks[i].setWeights(optimal_instance.getData())

            start = time.time()
            for instance in instances:
                networks[i].setInputValues(instance.getData())
                networks[i].run()

                predicted = instance.getLabel().getContinuous()
                actual = networks[i].getOutputValues().get(0)

                if abs(predicted - actual) < 0.5:
                    correct += 1
                else:
                    incorrect += 1

            end = time.time()
            testing_time = end - start

            results += "\nResults for %s: \nCorrectly classified %d instances." % (name, correct)
            accuracy1 = float(correct)/(correct+incorrect)*100.0
            results += "\nIncorrectly classified %d instances.\nPercent correctly classified: %0.03f%%" % (incorrect, accuracy1)
            results += "\nTraining time: %0.03f seconds" % (training_time,)
            results += "\nTesting time: %0.03f seconds\n" % (testing_time,)
            accuracies[i].append(accuracy1)
            training_times[i].append(training_time)

        print results
        
    with open(OUTPUT_FILE, "w") as outFile:
        for i in range(1):
            outFile.write(','.join([
                "iterations",
                "rhc_accuracy",
                "rhc_training_time",
                "sa_accuracy",
                "sa_training_time",
                "ga_accuracy",
                "ga_training_time"]) + '\n')
        for i in range(len(TRAINING_ITERATIONS_pool)):
            outFile.write(','.join([
                str(TRAINING_ITERATIONS_pool[i]),
                str(accuracies[0][i]),
                str(training_times[0][i]),
                str(accuracies[1][i]),
                str(training_times[1][i]),
                str(accuracies[2][i]),
                str(training_times[2][i])]) + '\n')
    print("the end of the program")
    
    

if __name__ == "__main__":
    main()

