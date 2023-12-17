import numpy as np

class Value:
    """An expression DAG node that stores fwd pass value and its gradient"""

    def __init__(self, data, label, children=()):
        self.data = data
        self.label = label
        self.children = children
        # This stores dFinalOutput / dThisNode after the entire bwd pass is complete.
        self.grad = 0
        # Increase all the children's grad by (dFinalOutput/dThisNode) * (dThisNode/dChild)
        # Should only be called after self.grad is set.
        self.backward = lambda: None

    def __mul__(self, other):
        c1 = self
        c2 = other
        out = Value(
            c1.data * c2.data,
            f"({c1.label} * {c2.label})",
            children=(c1, c2),
        )
        def backward():
            c1.grad += out.grad * c2.data
            c2.grad += out.grad * c1.data
        out.backward = backward
        return out

    def __add__(self, other):
        c1 = self
        c2 = other
        out = Value(
            c1.data + c2.data,
            f"({c1.label} + {c2.label})",
            children=(c1, c2),
        )
        def backward():
            c1.grad += out.grad * 1
            c2.grad += out.grad * 1
        out.backward = backward
        return out

    def tanh(self):
        e2x = np.exp(2 * self.data)
        out = Value(
            (e2x - 1) / (e2x + 1),
            f"tanh({self.label})",
            # Note need the comma to indicate it's a tuple w/ a single element.
            children=(self,),
        )
        def backward():
            self.grad += out.grad * (1 - out.data ** 2)
        out.backward = backward
        return out


    def __repr__(self):
        return f"{self.label} | data: {self.data:.4f} | grad: {self.grad:.4f}"
