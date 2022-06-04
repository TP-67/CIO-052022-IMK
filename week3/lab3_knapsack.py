"""
Tutorial:
https://pub.towardsai.net/genetic-algorithm-ga-introduction-with-example-code-e59f9bc58eaf

https://towardsdatascience.com/introduction-to-optimization-with-genetic-algorithm-2f5001d9964b
https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6
https://towardsdatascience.com/artificial-neural-networks-optimization-using-genetic-algorithm-with-python-1fe8ed17733e

https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35

Github:
https://github.com/Lucretiel/genetics
https://github.com/rmsolgi/geneticalgorithm

https://github.com/harvitronix/neural-network-genetic-algorithm
https://github.com/ahmedfgad/GeneticAlgorithmPython
"""

import numpy as np
import matplotlib.pyplot as plt

import copy
import itertools
from typing import List, Tuple, Dict


# Set random seed
np.random.seed(2022)

# Fitness function
def knapsack_fitness_func(chromosome: np.array,
                          weight_list: np.array,
                          value_list: np.array,
                          weight_limitation: int) -> float:
    """
    Fitness: An evaluator of each chromosome.
    The fitness function return a fitness score to each chromosome.
    """

    weight_total = weight_list * chromosome
    value_total = value_list * chromosome

    if weight_total.sum() > weight_limitation:
        return 0
    else:
        return value_total.sum()


def roulette_wheel_prob(population: np.array, beta: float) -> np.array:
    """
    Calculating probability for roulette wheel selection.
    """
    score = []
    for i in range(len(population)):
        score.append(population[i]['score'])

    # (N,)
    score = np.array(score)
    avg_score = np.mean(score)
    if avg_score != 0:
        score = score / avg_score

    # (N,)
    return np.exp(-beta * score)


def roulette_wheel_selection(prob: np.array) -> np.array:
    """
    Parents: two selected chromosome for generating the next generation.
    The chromosome with higher fitness score has bigger chance being selected for reproduction.
    We take the cumsum of probabilities and select the first parent whose cumsum is greater than random number.
    """

    # (N,)
    c = np.cumsum(prob)
    r = sum(prob) * np.random.rand()
    # (M,)
    ind = np.argwhere(r <= c)

    return ind[0][0]


# Crossover
def one_point_crossover(parent1: np.array, parent2: np.array) -> Tuple:
    """
    Introduce randomness into the population by swapping parents' chromosomes.
    Randomly select a crossover point for a pair of parents and exchange the genes of parents.
    """
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    # One-point crossover
    one_point_index = np.random.randint(1, len(parent1['gene'] - 1))

    child1['gene'] = np.concatenate((parent1['gene'][: one_point_index], parent2['gene'][one_point_index:]))
    child2['gene'] = np.concatenate((parent2['gene'][: one_point_index], parent1['gene'][one_point_index:]))

    return child1, child2


# Mutation
def binary_mutation(chromosome: np.array, mu: float) -> np.array:
    """
    The genes in a new formed offspring subject to a mutation with a low probability.
    Mutation maintains diversity within a population preventing the population from early convergence.

    Args:
        chromosome: Child chromosome.
        mu: Mutation rate. % of gene to be modified.
    """
    y = copy.deepcopy(chromosome)
    flag = np.random.rand(*chromosome['gene'].shape) <= mu
    ind = np.argwhere(flag)

    for i in ind[:, 0]:
        if y['gene'][i] == 1:
            y['gene'][i] = 0
        else:
            y['gene'][i] = 1

    return y


def uniform_mutation(chromosome: np.array, mu: float, sigma: float) -> np.array:
    """
    Args:
        chromosome: child chromosome.
        mu: mutation rate. % of gene to be modified
        sigma: step size of mutation
    """

    y = copy.deepcopy(chromosome)
    flag = np.random.rand(*chromosome['gene'].shape) <= mu
    ind = np.argwhere(flag)

    y['position'][ind] = y['position'][ind] + sigma * np.random.randn(*ind.shape)
    y['position'][ind] = [0 for _ in range(len(ind))] if y['position'][ind] == 1 else 0

    return y


def swap_mutation(chromosome: np.array, mu: float) -> np.array:
    pass


def sort_chromosome(population: np.array, population_size: int) -> np.array:
    """
    Bubble sorting the population and offspring in every iteration to get best fit individuals at top.
    Add new formed offsprings to the population.
    The chromosomes with lower fitness scores are deleted from the population as the offspring formed.
    """
    n = len(population)

    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if population[j]['score'] < population[j + 1]['score']:
                population[j], population[j + 1] = population[j + 1], population[j]

    return dict(itertools.islice(population.items(), population_size))


# Initial population
def binary_initialization(population_size: int,
                          gene_num: int,
                          weight_list: np.array,
                          value_list: np.array,
                          weight_limitation: int) -> Dict:
    """
    Gene: an element of the problem.
    Individual (chromosome): a solution that satisfies restrictions.
    Population: a set of possible chromosomes.
    """

    population = {}
    # Each individual has position(chromosomes) and cost
    for i in range(population_size):
        population[i] = {'gene': None, 'score': None}

    # First generation
    for i in range(population_size):
        # Randomly initialize chromosomes
        population[i]['gene'] = np.random.randint(2, size=gene_num)

        # Calculate fitness scores
        population[i]['score'] = knapsack_fitness_func(population[i]['gene'],
                                                       weight_list,
                                                       value_list,
                                                       weight_limitation)

    return population


def knapsack_evaluation(population_size: int,
                        gene_num: int,
                        epoch: int,
                        beta: float,
                        mu: float,
                        weight_list: np.array,
                        value_list: np.array,
                        weight_limitation: int):
    population = binary_initialization(population_size,
                                       gene_num,
                                       weight_list,
                                       value_list,
                                       weight_limitation)

    # Define best chromosome in the first generation
    # best_score = np.inf
    # best_chromosome = None
    #
    # for i in range(population_size):
    #     if population[i]['score'] < best_score:
    #         best_chromosome = copy.deepcopy(population[i])
    #         best_score = population[i]['score']

    # Find best chromosome during iteration
    best_score_list = []
    best_chromosome_list = []

    # Main loop
    for iteration in range(epoch):
        prob = roulette_wheel_prob(population, beta)

        for _ in range(population_size // 2):
            # Roulette wheel selection
            p1 = population[roulette_wheel_selection(prob)]
            p2 = population[roulette_wheel_selection(prob)]

            # crossover two parents
            c1, c2 = one_point_crossover(p1, p2)

            # Perform mutation
            c1 = binary_mutation(c1, mu)
            c2 = binary_mutation(c2, mu)

            score_c1 = knapsack_fitness_func(c1['gene'],
                                             weight_list,
                                             value_list,
                                             weight_limitation)
            score_c2 = knapsack_fitness_func(c2['gene'],
                                             weight_list,
                                             value_list,
                                             weight_limitation)

            c1['score'] = score_c1
            c2['score'] = score_c2

            # best_score = max(score_c1, score_c2)
            # best_chromosome = copy.deepcopy(c1) if score_c1 > score_c2 else copy.deepcopy(c2)

            # Merge, Sort and Select
            population[len(population)] = c1
            population[len(population)] = c2

        # The algorithm is terminated when offspring are similar to the previous generation (converged).
        population = sort_chromosome(population, population_size)

        # Store best cost
        # best_score_list.append(best_score)
        # best_chromosome_list.append(best_chromosome)
        best_score_list.append(population[0]['score'])
        best_chromosome_list.append(population[0]['gene'])

        # Show generation information
        print('Iteration {}: Best Cost = {}: Best Solution = {}'.format(iteration,
                                                                        best_score_list[iteration],
                                                                        best_chromosome_list[iteration]))

    return best_score_list, best_chromosome_list


if __name__ == '__main__':
    weight_list = np.array([9, 13, 153, 50, 15, 68, 27, 39, 23, 52, 11, 32, 24, 48, 73, 42, 43, 22, 7, 18, 4, 30])
    value_list = np.array([150, 35, 200, 160, 60, 45, 60, 40, 30, 10, 70, 30, 15, 10, 40, 70, 75, 80, 20, 12, 50, 10])

    best_cost, best_solution = knapsack_evaluation(population_size=20,
                                                   gene_num=22,
                                                   epoch=5001,
                                                   beta=1.0,
                                                   mu=0.1,
                                                   weight_list=weight_list,
                                                   value_list=value_list,
                                                   weight_limitation=400)

    plt.plot(best_cost)
    plt.xlim(0, 5001)
    plt.xlabel('Generations')
    plt.ylabel('Best Cost')
    plt.title('Genetic Algorithm')
    plt.grid(True)
    plt.show()
