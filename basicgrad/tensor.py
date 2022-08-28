import math
from basicgrad.traceGraph import draw_dot


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self._prev = _children
        self._op = _op
        self.grad = 0
        self._backward = lambda: None
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}{f' label={self.label}' if self.label else ''})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        # for add we just propogate the grads
        def _backward():
            # += cuz of accumulation of grads
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "+")
        # for add we multiply the grads by the opp value
        def _backward():
            # += cuz of accumulation of grads
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * self**-1

    def tanh(self):
        x = self.data
        val = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(val, (self,), "tanh")
        # derivative of : dtanh(x)/dx = 1 - tanh(x)**2
        def _backward():
            # += cuz of accumulation of grads
            self.grad += (1 - val**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        x = self.data
        val = max(0, x)
        out = Value(val, (self,), "relu")

        def _backward():
            # += cuz of accumulation of grads
            self.grad += (0 if out.data <= 0 else 1) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def exp(self):
        x = self.data
        val = math.exp(x)
        out = Value(val, (self,), "exp")

        def _backward():
            self.grad += val * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # Topological Sort
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
        return self

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supports int, float"  # TODO: Try to expand by refering to pytorch code
        out = Value(self.data**other, (self,), "*")
        # for add we multiply the grads by the opp value
        def _backward():
            # += cuz of accumulation of grads
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def summary(self):
        return draw_dot(self)
