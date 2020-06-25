import numpy as np
from .input_handler import mapping


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def mse(y, y_pred):
    return 0.5 * np.mean(np.power(y - y_pred, 2))


class Taskset:
    '''Including:
        - a dataset
        - three ANN configuration
    '''

    def __init__(self, config):
        self.config = config
        self.X, self.y = mapping[self.config['name']](self.config['path'])

    @property
    def D_multitask(self):
        num_input = self.config['input']
        num_hidden_max = max(self.config['hiddens'])
        return (num_input + 1) * num_hidden_max + num_hidden_max + 1

    @property
    def dims(self):
        num_input = self.config['input']
        hidden = self.config['hiddens']
        d = []
        for h in range(len(hidden)):
            n_h = hidden[h]
            d.append((num_input + 1) * n_h + n_h + 1)
        return d

    @property
    def dim(self):
        num_input = self.config['input']
        hidden = self.config['hiddens']
        n_h = hidden[0]
        return (num_input + 1) * n_h + n_h + 1

    def indirect_decode(self, solution, sf):
        num_input = self.config['input']
        num_hidden = self.config['hiddens'][sf]
        num_hidden_max = max(self.config['hiddens'])
        assert len(solution) == self.D_multitask
        start = 0
        end = start + num_input * num_hidden_max
        w1 = solution[start:end].reshape(num_input, num_hidden_max)[:, :num_hidden]
        w1 = w1 * 10 - 5

        start = end
        end = start + num_hidden_max
        b1 = solution[start:end][:num_hidden]
        b1 = b1 * 10 - 5

        start = end
        end = start + num_hidden_max
        w2 = solution[start:end].reshape(num_hidden_max, 1)[:num_hidden, :]
        w2 = w2 * 10 - 5

        start = end
        end = start + 1
        b2 = solution[start:end]
        b2 = b2 * 10 - 5
        return w1, b1, w2, b2

    def decode_pop_to_task_size(self, subpops):
        new_sub_pop = []
        dims = []
        num_input = self.config['input']
        num_hidden_max = max(self.config['hiddens'])
        for sf in range(len(subpops)):
            pop = subpops[sf]
            num_hidden = self.config['hiddens'][sf]
            start = 0
            end = start + num_input * num_hidden_max
            w1 = np.arange(start, start + num_input * num_hidden)
            w1_ = np.arange(start + num_input * num_hidden, start + num_input * num_hidden_max)
            start = end
            end = start + num_hidden_max
            b1 = np.arange(start, start + num_hidden)
            b1_ = np.arange(start + num_hidden, start + num_hidden_max)
            start = end
            end = start + num_hidden_max
            w2 = np.arange(start, start + num_hidden)
            w2_ = np.arange(start + num_hidden, start + num_hidden_max)
            start = end
            end = start + 1
            b2 = np.arange(start, end)
            idx = np.concatenate([w1, b1, w2, b2, w1_, b1_, w2_])
            pop = pop[:, idx]
            new_sub_pop.append(pop)
        return new_sub_pop

    def direct_decode(self, solution):
        num_input = self.config['input']
        num_hidden = self.config['hidden']
        # num_hidden_max = max(self.config['hiddens'])
        assert len(solution) == self.D_multitask
        start = 0
        end = start + num_input * num_hidden
        w1 = solution[start:end].reshape(num_input, num_hidden)
        w1 = w1 * 10 - 5

        start = end
        end = start + num_hidden
        b1 = solution[start:end][:num_hidden]
        b1 = b1 * 10 - 5

        start = end
        end = start + num_hidden
        w2 = solution[start:end].reshape(num_hidden, 1)
        w2 = w2 * 10 - 5

        start = end
        end = start + 1
        b2 = solution[start:end]
        b2 = b2 * 10 - 5
        return w1, b1, w2, b2

    def evaluate(self, solution):
        """
        Params
        ------
        - solution (vector): vector of weights of ANN
        """
        w1, b1, w2, b2 = self.direct_decode(solution)
        out = sigmoid(self.X @ w1 + b1)
        out = sigmoid(out @ w2 + b2)
        return mse(self.y, out)


if __name__ == '__main__':
    pass
