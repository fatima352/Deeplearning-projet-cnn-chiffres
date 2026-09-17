import numpy as np


class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(0, x)


if __name__ == "__main__":

    x = np.array([
        [-2, 3, -1],
        [4, -5, 6]
    ])

    relu = ReLU()

    output = relu.forward(x)

    print("Entrée :")
    print(x)

    print("Sortie ReLU :")
    print(output)