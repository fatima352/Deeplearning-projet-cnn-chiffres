import numpy as np


class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(0, x)


class Linear:
    def __init__(self, n_inputs, n_outputs):
        self.weights = np.random.randn(n_outputs, n_inputs)
        self.bias = np.zeros(n_outputs)

    def forward(self, x):
        return np.matmul(self.weights, x) + self.bias


if __name__ == "__main__":

    # =========================
    # Test ReLU
    # =========================
    print("=== Test ReLU ===")

    x_relu = np.array([
        [-2, 3, -1],
        [4, -5, 6]
    ])

    relu = ReLU()
    output_relu = relu.forward(x_relu)

    print("Entrée :")
    print(x_relu)

    print("Sortie ReLU :")
    print(output_relu)


    # =========================
    # Test Linear
    # =========================
    print("\n=== Test Linear ===")

    x_linear = np.array([2.0, 5.0, 1.0])

    linear = Linear(n_inputs=3, n_outputs=2)
    output_linear = linear.forward(x_linear)

    print("Entrée :", x_linear)
    print("Shape entrée :", x_linear.shape)

    print("\nPoids :")
    print(linear.weights)
    print("Shape poids :", linear.weights.shape)

    print("\nBiais :", linear.bias)
    print("Shape biais :", linear.bias.shape)

    print("\nSortie :", output_linear)
    print("Shape sortie :", output_linear.shape)