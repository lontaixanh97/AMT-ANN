import numpy as np
from scipy.stats import multivariate_normal


class ProbabilityModel:  # Works reliably for 2(+) Dimensional distributions
    """ properties
        modeltype; % multivariate normal ('mvarnorm' - for real coded) or univariate marginal distribution ('umd' - for binary coded)
        mean_noisy;
        mean_true;
        covarmat_noisy;
        covarmat_true;
        probofone_noisy;
        probofone_true;
        probofzero_noisy;
        probofzero_true;
        vars;
      end"""

    def sample(self, nos):
        # print('nos,self.vars', nos,self.vars)
        nos = int(nos)
        solutions = np.random.rand(nos, int(self.vars))
        for i in range(nos):
            index1 = solutions[i, :] <= self.probofone_true
            index0 = solutions[i, :] > self.probofone_true
            solutions[i, index1] = 1
            solutions[i, index0] = 0
        return solutions

    def pdfeval(self, solutions):
        """Calculating the probabilty of every solution
            Tính phân phối xác suất cho mỗi solution
        Arguments:
            solutions {[2-D Array]} -- [solution or population of evolutionary algorithm]
            Đầu vào, solutions 2-D array, quần thể của EA
        Returns:
            [1-D Array] -- [probabilty of every solution]

        """
        nos = solutions.shape[0]
        probofsols = np.zeros(nos)
        probvector = np.zeros(self.vars)
        for i in range(nos):
            index = solutions[i, :] == 1
            probvector[index] = self.probofone_noisy[index]
            index = solutions[i, :] == 0
            probvector[index] = self.probofzero_noisy[index]
            probofsols[i] = np.prod(probvector)
        return probofsols

    def buildmodel(self, solutions):
        pop, self.vars = solutions.shape

        self.probofone_true = np.mean(solutions, 0)
        # print(self.probofone_true)
        self.probofzero_true = 1 - self.probofone_true
        # print('probofone_true')
        # print(self.probofzero_true.shape)
        solutions_noisy = np.append(solutions, np.round(np.random.rand(round(0.1 * pop), self.vars)), axis=0)
        # print(solutions_noisy.shape)
        self.probofone_noisy = np.mean(solutions_noisy, 0)
        # print(self.probofone_noisy)
        self.probofzero_noisy = 1 - self.probofone_noisy
        # print(self.probofzero_noisy)
