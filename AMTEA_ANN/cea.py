from mfea_ii_lib import *
from utils.tools import *
from probability_model import ProbabilisticModel


def cea(taskset, config, buildModel, path, callback=None):
    num_input = taskset.config['input']
    N = config['pop_size']  # size of population
    D = taskset.dim  # max dim

    T = config['num_iter']  # num iterator
    sbxdi = config['sbxdi']
    pmdi = config['pmdi']
    pswap = config['pswap']

    # initialize
    population = np.random.rand(2 * N, D)  # init population with size = size *2; dim = dim max
    fitness = np.full(2 * N, np.inf)
    bestSol = None
    all_models = Tools.load_from_file(os.path.join('problems/', path))
    fitness_hist = np.zeros(T)

    # evaluate
    for i in range(2 * N):
        fitness[i] = taskset.evaluate(population[i])

    # sort]
    sort_index = np.argsort(fitness)
    population = population[sort_index]
    fitness = fitness[sort_index]
    fitness_hist[0] = fitness[0]

    # evolve
    iterator = trange(T)
    for t in iterator:
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

        fitness_hist[t] = fitness[0]

        # optimization info
        message = {'algorithm': 'cea'}
        result = get_optimize_results(t, population, fitness, message)
        if callback:
            callback(result)

        if config['is_test']:
            desc = 'gen:{} fitness:{} message:{}'.format(t, ''.join(
                '{:0.4f}'.format(result.fun)), message)
            iterator.set_description(desc)
    bestSol = population[0, :]

    if buildModel:
        model = ProbabilisticModel('mvarnorm')
        model.buildModel(solutions=population, num_input=num_input)
        all_models.append(model)
        Tools.save_to_file(os.path.join('problems/', path), all_models)

    return bestSol, fitness_hist
