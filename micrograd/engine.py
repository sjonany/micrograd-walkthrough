import numpy as np

class Value:
    """An expression DAG node that stores fwd pass value and its gradient"""

    def __init__(self, data, label='', children=()):
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
        c2 = self.__ensure_value(other)
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
        c2 = self.__ensure_value(other)
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
    
    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other
    
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

    def exp(self):
        out = Value(
            np.exp(self.data),
            f"e^({self.label})",
            children=(self,),
        )
        def backward():
            self.grad += out.grad * out.data
        out.backward = backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(
            self.data ** other,
            f"{self.label}^({other})",
            children=(self,),
        )
        def backward():
            self.grad += out.grad * other * self.data ** (other -1)
        out.backward = backward
        return out

    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)

    def run_full_backpropagation(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Reset gradients before all the grad +='s
        for v in topo:
            v.grad = 0

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v.backward()

    def __repr__(self):
        return f"{self.label} | data: {self.data:.4f} | grad: {self.grad:.4f}"
    
    @staticmethod
    def __ensure_value(v):
        return v if isinstance(v, Value) else Value(v, str(v))
