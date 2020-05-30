import numpy as np
from scipy.stats import multivariate_normal
"""This is to build a probabilistic model and mixture model for the population."""


class ProbabilisticModel(object):
    def __init__(self, modelType):
        self.num_input = None
        self.type = modelType
        self.dim = None
        self.sol = None
        if self.type == 'mvarnorm':
            self.mean = None
            self.cov = None
            self.mean_noisy = None
            self.cov_noisy = None
        else:
            raise ValueError('Invalid probabilistic model type!')

    def buildModel(self, solutions, num_input):
        self.num_input = num_input
        self.sol = solutions
        pop = solutions.shape[0]
        self.dim = solutions.shape[1]
        if self.type == 'mvarnorm':
            self.mean = np.mean(solutions, axis=0)
            self.cov = np.cov(solutions.T)
            self.cov = np.diag(np.diag(self.cov))
            solutions_noisy = np.concatenate((solutions, np.random.rand(int(0.1 * pop), self.dim)), axis=0)
            self.mean_noisy = np.mean(solutions_noisy, axis=0)
            self.cov_noisy = np.cov(solutions_noisy.T)
            self.cov_noisy = np.diag(np.diag(self.cov_noisy))

    def sample(self, nSol):
        if self.type == 'mvarnorm':
            solutions = np.random.multivariate_normal(self.mean, self.cov, nSol)
        return solutions

    def pdfEval(self, solutions):
        if self.type == 'mvarnorm':
            pdf = multivariate_normal.pdf(solutions, mean=self.mean_noisy, cov=self.cov_noisy)
        return pdf

    def gen_decode(self, num_hidden, max_hidden, solution):
        max_dim = (self.num_input + 1)*max_hidden + max_hidden + 1
        tmp = np.mean(solution)
        pop_size = solution.shape[0]
        new_solution = []
        for i in range(pop_size):
            gen = solution[i]
            new_gen = []
            # them thanh phan w1 vao new_gen
            start = 0
            end = start + self.num_input * num_hidden
            new_gen = np.append(new_gen, gen[start:end])

            # them vao nhung khoang trong gia tri bang trung binh cua gen
            new_gen = np.append(new_gen, tmp * np.ones(self.num_input * (max_hidden - num_hidden)), )

            # them thanh phan b1 vao new_gen
            start = end
            end = start + num_hidden
            new_gen = np.append(new_gen, gen[start:end])

            # them vao nhung khoang trong gia tri bang trung binh cua gen
            new_gen = np.append(new_gen, tmp * np.ones(max_hidden - num_hidden), )

            # them thanh phan w2 vao new_gen
            start = end
            end = start + num_hidden
            new_gen = np.append(new_gen, gen[start:end])

            # them vao nhung khoang trong gia tri bang trung binh cua gen
            new_gen = np.append(new_gen, tmp * np.ones(max_hidden - num_hidden), )

            # them thanh phan b2 vao new_gen
            start = end
            end = start + 1
            new_gen = np.append(new_gen, gen[start:end])

            new_solution = np.concatenate((new_solution, new_gen), axis=0)

        return new_solution.reshape(pop_size, max_dim)

    def indirect_decode(self, num_hidden, max_hidden, solution):
        pop_size = solution.shape[0]
        new_solution = []
        dim = (self.num_input + 1) * num_hidden + num_hidden + 1
        for i in range(pop_size):
            gen = solution[i]
            new_gen = []
            start = 0
            end = start + self.num_input * max_hidden
            new_gen = np.append(new_gen, gen[start:end].reshape(self.num_input, max_hidden)[:, :num_hidden])

            start = end
            end = start + max_hidden
            new_gen = np.append(new_gen, gen[start:end][:num_hidden])

            start = end
            end = start + max_hidden
            new_gen = np.append(new_gen, gen[start:end].reshape(max_hidden, 1)[:num_hidden, :])

            start = end
            end = start + 1
            new_gen = np.append(new_gen, gen[start:end])
            new_solution = np.concatenate((new_solution, new_gen), axis=0)

        return new_solution.reshape(pop_size, dim)

    def modify1(self, dims):
        num_hiddens = int((dims - 1) / (self.num_input + 2))
        num_hidden = int((self.dim - 1) / (self.num_input + 2))
        if dims < self.dim:
            new_sol = self.indirect_decode(num_hiddens, num_hidden, self.sol)
            self.buildModel(new_sol, self.num_input)
        elif dims > self.dim:
            new_sol = self.gen_decode(num_hidden, num_hiddens, self.sol)
            self.buildModel(new_sol, self.num_input)
        self.dim = dims

    def modify(self, dims):
        if dims < self.dim:
            if self.type == 'mvarnorm':
                self.mean = self.mean[:dims]
                self.cov = self.cov[:dims, :dims]
                self.mean_noisy = self.mean_noisy[:dims]
                self.cov_noisy = self.cov_noisy[:dims, :dims]

        elif dims > self.dim:
            if self.type == 'mvarnorm':
                mean_cat = np.zeros(dims) + 0.5
                mean_cat[:self.dim] = self.mean
                self.mean = mean_cat
                cov_cat = np.diag(np.ones(dims) + 1)
                cov_cat[:self.dim, :self.dim] = self.cov
                self.cov = cov_cat

                mean_cat = np.zeros(dims) + 0.5
                mean_cat[:self.dim] = self.mean_noisy
                self.mean_noisy = mean_cat
                cov_cat = np.diag(np.ones(dims))
                cov_cat[:self.dim, :self.dim] = self.cov_noisy
                self.cov_noisy = cov_cat
        self.dim = dims
