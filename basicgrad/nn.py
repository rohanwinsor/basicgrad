import random
from .tensor import Value


class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1), label=f"w{i}") for i in range(nin)]
        self.b = Value(random.uniform(-1, 1), label="b")
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum([wi * xi for wi, xi in zip(self.w, x)], self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class MSELoss:
    def __call__(self, outs, preds):
        losss = []
        for out, pred in zip(outs, preds):
            loss = (out - pred) ** 2
            losss.append(loss)
        loss = sum(losss)
        return loss / len(losss)
