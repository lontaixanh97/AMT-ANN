import os
import numpy as np

from utils.tools import *
from utils.fitness_util import *
from probability_model import *
from mixture_model import *


def AMT_BGA(problem, dims, reps, trans, addr="problems/knapsack"):
    """[bestSol, fitness_hist, alpha] = TSBGA(problem, dims, reps, trans): Adaptive
        Model-based Transfer Binary GA. The crossover and mutation for this simple
        binary GA are uniform crossover and bit-flip mutation.
        INPUT:
         problem: problem type, 'KP_uc_rk', 'KP_sc_rk', 'KP_wc_rk' or 'KP_sc_ak'
         dims: problem dimensionality
         reps: number of repeated trial runs
         trans:    trans.transfer: binary variable
                   trans.TrInt: transfer interval for AMT

        OUTPUT:
         bestSol: best solution for each repetiion
         fitness: history of best fitness for each generation
         alpha: transfer coefficient
    """
    pop = 50
    gen = 1000
    transfer = trans['transfer']
    if transfer:
        TrInt = trans['TrInt']
        all_models = Tools.load_from_file(os.path.join(addr, 'all_models'))

    fitness_hist = np.zeros((reps, gen))
    bestSol = np.zeros((reps, dims))
    alpha = [None] * (reps)

    for rep in range(reps):
        alpha_rep = []
        population = np.round(np.random.rand(pop, dims))
        fitness = knapsack_fitness_eval(population, problem, dims, pop)
        ind = np.argmax(fitness)
        best_fit = fitness[ind]
        print('Generation 0 best fitness = ', str(best_fit))
        fitness_hist[rep, 0] = best_fit

        for i in range(1, 10):
            # As we consider all the population as parents, we don't samplt P^{s}
            if transfer and i % TrInt == 0:
                mmodel = MixtureModel(all_models)
                mmodel.createtable(population, True)
                mmodel.EMstacking()  # Recombination of probability models
                mmodel.mutate()  # Mutation of stacked probability model
                offspring = mmodel.sample(pop)
                alpha_rep.append(mmodel.alpha)
                print('Transfer coefficient at generation ', str(i), ': ', str(mmodel.alpha))

            else:
                parent1 = population[np.random.permutation(pop), :]
                parent2 = population[np.random.permutation(pop), :]
                tmp = np.random.rand(pop, dims)
                offspring = np.zeros((pop, dims))
                index = tmp >= 0.5
                offspring[index] = parent1[index]
                index = tmp < 0.5
                offspring[index] = parent2[index]
                tmp = np.random.rand(pop, dims)
                index = tmp < (1 / dims)
                offspring[index] = np.abs(1 - offspring[index])

            cfitness = knapsack_fitness_eval(population, problem, dims, pop)
            interpop = np.append(population, offspring, 0)
            interfitness = np.append(fitness, cfitness)
            index = np.argsort((-interfitness))
            interfitness = interfitness[index]
            fitness = interfitness[:pop]
            interpop = interpop[index, :]
            population = interpop[:pop, :]
            print('Generation ', str(i), ' best fitness = ', str(np.max(fitness_hist)))
            fitness_hist[rep, i] = fitness[0]

        alpha[rep] = alpha_rep
        bestSol[rep, :] = population[ind, :]
    return bestSol, fitness_hist, alpha


def main():
    knapsack_problem_path = 'problems/knapsack'
    reps = 30
    TrInt = 2

    trans = {'transfer': True, 'TrInt': TrInt}
    KP_wc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_wc_ak'))
    bestSol, fitness_hit, alpha = AMT_BGA(KP_wc_ak, 1000, reps, trans, addr="problems/knapsack")


if __name__ == "__main__":
    main()
