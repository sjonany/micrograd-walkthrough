"""
Multi-layer perceptron
"""
import numpy as np
from .engine import Value
import pdb
class Neuron:
    """"A scalar function, f(x^) = w^.x^ + b, where size of w^ = n_in"""
    def __init__(self, n_in):
        """randomly init params with Uniform[-1,1]"""
        self.w = [Value(np.random.uniform(-1, 1), label=f"w{i}") for i in range (n_in)]
        self.b = Value(np.random.uniform(-1, 1), label="b")

    def __call__(self, x):
        """Evaluate. x is a vector of size n_in. Output is a Value"""
        return (sum([self.w[i] * x[i] for i in range(len(x))]) + self.b).tanh()


class Layer:
    """
    A list of neurons, each with its own weights.
    Intuitively, converts a vector into another vector by having each neuron execute independently
    """
    def __init__(self, n_in, n_out):
        """Create n_out neurons, each taking in vectors of size n_in."""
        self.neurons = [Neuron(n_in) for _ in range(n_out)]
    
    def __call__(self, x):
        """
        Evaluate.
        Input: x , a vector of size n_in
        Output: a vector of Values of size n_out
        """
        return [n(x) for n in self.neurons]


class MLP:
    """Multi-layer perceptron. A list of layers"""

    def __init__(self, n_in, n_outs):
        """
        A function that takes in a vector of size n_in, and outputs a vector of size n_outs[-1]
        n_outs determines the output dimension of intermediate layers.
        """
        dims = [n_in] + list(n_outs)
        self.layers = [Layer(dims[layer_i], dims[layer_i+1]) for layer_i in range(len(dims) - 1)]
        
        # Collect all the params so it's easy to do gradient descent
        self.parameters = []
        for layer in self.layers:
            for neuron in layer.neurons:
                self.parameters.extend(neuron.w)
                self.parameters.append(neuron.b)

    def __call__(self, x):
        """
        Evaluate. The interface is the same as Layer.__call__.
        Input: x, a vector of size n_in
        Output: a vector of Values of size n_outs[-1]
        """
        cur_vec = x
        for layer in self.layers:
            cur_vec = layer(cur_vec)
        return cur_vec
