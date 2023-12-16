class Value:
    """An expression DAG node that stores fwd pass value and its gradient"""

    def __init__(self, data, label, children=(), op=""):
        self.data = data
        self.label = label
        self.children = children
        self.op = op

    def __mul__(self, other):
        return Value(
            self.data * other.data,
            f"({self.label} * {other.label})",
            children=(self, other),
            op="*",
        )

    def __add__(self, other):
        return Value(
            self.data + other.data,
            f"({self.label} + {other.label})",
            children=(self, other),
            op="+",
        )

    def __repr__(self):
        return f"{self.label} | data: {self.data:.4f}"
