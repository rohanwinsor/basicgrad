import random
from .tensor import Value

class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1), label=f'w{i}') for i in range(nin)]
        self.b = Value(random.uniform(-1,1), label='b')
        # self.w = [Value(1.8)]
        # self.b = Value(32.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum([wi*xi for wi, xi in zip(self.w,x)], self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
