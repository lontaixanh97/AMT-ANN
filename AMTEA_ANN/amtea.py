import os
import numpy as np
from mfea_ii_lib import *
from probability_model import ProbabilisticModel
from mixture_model import MixtureModel
from utils.tools import *


def amtea(taskset, config, trans, buildmodel,  callback=None):
    transfer = trans['transfer']
    if transfer:
        TrInt = trans['TrInt']
        all_models = Tools.load_from_file(os.path.join('problems/', 'all_models'))

    N = config['pop_size']  # size of population
    D = taskset.dim  # dim

    T = config['num_iter']  # num iterator
    sbxdi = config['sbxdi']
    pmdi = config['pmdi']
    pswap = config['pswap']
    alpha_rep = []

    # initialize
    population = np.random.rand(2 * N, D)  # init population with size = size *2; dim = dim max
    fitness = np.full(2 * N, np.inf)
    bestfit_hist = np.zeros(T)

    # evaluate
    for i in range(2 * N):
        fitness[i] = taskset.evaluate(population[i])

    # sort]
    sort_index = np.argsort(fitness)
    population = population[sort_index]
    fitness = fitness[sort_index]
    bestfit_hist[0] = fitness[0]
    bestSol = population[0, :]

    # evolve
    iterator = trange(T)
    for t in iterator:
        if trans['transfer'] and t % trans['TrInt'] == 0:
            mixModel = MixtureModel(all_models)
            mixModel.createTable(population, True, 'mvarnorm', D)
            mixModel.EMstacking()
            mixModel.mutate()
            offspring = mixModel.sample(N)
            population[N::, :] = offspring
            alpha_rep = np.concatenate((alpha_rep, mixModel.alpha), axis=0)

        else:
            # permute current population
            permutation_index = np.random.permutation(N)
            population[:N] = population[:N][permutation_index]
            fitness[:N] = fitness[:N][permutation_index]
            fitness[N:] = np.inf

            # select pair to crossover
            for i in range(0, N, 2):
                # extract parent
                p1 = population[i]
                p2 = population[i + 1]
                # recombine parent
                c1, c2 = sbx_crossover(p1, p2, sbxdi)
                c1 = mutate(c1, pmdi)
                c2 = mutate(c2, pmdi)
                c1, c2 = variable_swap(c1, c2, pswap)
                # save child
                population[N + i, :], population[N + i + 1, :] = c1[:], c2[:]

        # evaluate
        for i in range(N, 2 * N):
            fitness[i] = taskset.evaluate(population[i])

        # sort
        sort_index = np.argsort(fitness)

        population = population[sort_index]
        fitness = fitness[sort_index]

        bestfit_hist[t] = fitness[0]
        bestSol = population[0, :]

        # optimization info
        message = {'algorithm': 'amtea'}
        result = get_optimize_results(t, population, fitness, message)
        if callback:
            callback(result)

        if config['is_test']:
            desc = 'gen:{} fitness:{} message:{}'.format(t, ''.join(
                '{:0.4f}'.format(result.fun)), message)
            iterator.set_description(desc)
    if buildmodel:
        model = ProbabilisticModel('mvarnorm')
        model.buildModel(solutions=population)
        all_models.append(model)
        Tools.save_to_file(os.path.join('problems/', 'all_models'), all_models)

    return bestSol, bestfit_hist
