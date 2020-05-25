import numpy as np
from scipy.stats import multivariate_normal
"""This is to build a probabilistic model and mixture model for the population."""


class ProbabilisticModel(object):
    def __init__(self, modelType):
        self.type = modelType
        self.dim = None
        if self.type == 'mvarnorm':
            self.mean = None
            self.cov = None
            self.mean_noisy = None
            self.cov_noisy = None
        else:
            raise ValueError('Invalid probabilistic model type!')

    def buildModel(self, solutions):
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
