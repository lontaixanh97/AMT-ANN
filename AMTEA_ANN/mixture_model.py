import numpy as np
from probability_model import ProbabilisticModel


class MixtureModel(object):
    def __init__(self, allModels):
        self.model_list = allModels.copy()
        self.nModels = len(allModels)
        self.alpha = (1 / self.nModels) * np.ones(self.nModels)
        self.probTable = None
        self.nSol = None

    def createTable(self, solutions, CV, num_input, modelType, dims, probs_RL=None):
        if CV:
            self.nModels = self.nModels + 1
            self.model_list.append(ProbabilisticModel(modelType=modelType))
            self.model_list[-1].buildModel(solutions, num_input)
            self.alpha = (1 / self.nModels) * np.ones(self.nModels)
            nSol = solutions.shape[0]
            self.nSol = nSol
            self.probTable = np.ones([nSol, self.nModels])
            if probs_RL is None:
                for j in range(self.nModels - 1):
                    # print(j)
                    self.model_list[j].modify1(dims)
                    self.probTable[:, j] = self.model_list[j].pdfEval(solutions)
                    # print(self.probTable.shape)
            else:
                for j in range(0, self.nModels - 2):
                    self.probTable[:, j] = self.model_list[j].pdfEval(solutions)
                self.probTable[:, -2] = probs_RL
            for i in range(nSol):  # Leave-one-out cross validation
                x = np.concatenate((solutions[:i, :], solutions[i + 1:, :]))
                tModel = ProbabilisticModel(modelType=modelType)
                tModel.buildModel(x, num_input)
                self.probTable[i, -1] = tModel.pdfEval(solutions[[i], :])
        else:
            nSol = solutions.shape[0]
            self.probTable = np.ones([nSol, self.nModels])
            for j in range(self.nModels):
                self.probTable[:, j] = self.model_list[j].pdfEval(solutions)
            self.nSol = nSol

    def EMstacking(self):
        iterations = 100
        for _ in range(iterations):
            talpha = self.alpha
            probVector = np.matmul(self.probTable, talpha.T)
            for i in range(self.nModels):
                talpha[i] = np.sum((1 / self.nSol) * talpha[i] * self.probTable[:, i] / probVector)
            self.alpha = talpha

    def mutate(self):
        modif_alpha = np.maximum(self.alpha + np.random.normal(0, 0.01, self.nModels), 0)
        total_alpha = np.sum(modif_alpha)
        if total_alpha == 0:
            self.alpha = np.zeros(self.nModels)
            self.alpha[-1] = 1
        else:
            self.alpha = modif_alpha / total_alpha

    def sample(self, nSol, samplesRL=None):
        indSamples = np.ceil(nSol * self.alpha).astype(int)
        solutions = np.array([])
        for i in range(self.nModels):
            if indSamples[i] == 0:
                pass
            elif i == self.nModels - 2 and samplesRL is not None:
                solutions = np.vstack([solutions, samplesRL]) if solutions.size else samplesRL
            else:
                sols = self.model_list[i].sample(indSamples[i])
                solutions = np.vstack([solutions, sols]) if solutions.size else sols
        solutions = solutions[np.random.permutation(solutions.shape[0]), :]
        solutions = solutions[:nSol, :]
        return solutions

    def n_samples(self, ind, nSol):
        return np.ceil(nSol * self.alpha[ind]).astype(int)
