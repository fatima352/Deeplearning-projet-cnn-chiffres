import numpy as np


class ReLU:
    def __init__(self):
        pass

    # Remplace les valeurs négatives par 0
    def forward(self, x):
        return np.maximum(0, x)


class Linear:
    def __init__(self, n_inputs, n_outputs):
        # Initialisation des poids de la couche
        self.weights = np.random.randn(n_outputs, n_inputs)
        # Initialisation des biais à 0
        self.bias = np.zeros(n_outputs)

    def forward(self, x):
        # Transformation linéaire : y = W*x + b
        return np.matmul(self.weights, x) + self.bias


class Conv2D:
    def __init__(self, n_filters, kernel_size):
        # Nombre de filtres et taille des kernels
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        # Initialisation des kernels avec des valeurs aléatoires
        self.kernels = np.random.randn(
            n_filters,
            kernel_size,
            kernel_size
        )

    def forward(self, x):
        # Récupération des dimensions de l'image d'entrée
        height, width = x.shape

        # Calcul des dimensions de la sortie
        output_height = height - self.kernel_size + 1
        output_width = width - self.kernel_size + 1

        # Création des feature maps de sortie
        output = np.zeros(
            (self.n_filters, output_height, output_width)
        )

        # Application de chaque filtre
        for f in range(self.n_filters):

            # Déplacement du filtre sur toute l'image
            for i in range(output_height):
                for j in range(output_width):

                    # Extraction de la zone de l'image sous le kernel
                    region = x[
                        i:i + self.kernel_size,
                        j:j + self.kernel_size
                    ]

                    # Convolution : multiplication puis somme
                    output[f, i, j] = np.sum(
                        region * self.kernels[f]
                    )

        # Retourne les feature maps obtenues
        return output

class MaxPool2D:
    def __init__(self, pool_size=2):
        # Taille de la fenêtre de pooling
        self.pool_size = pool_size

    def forward(self, x):
        # Dimensions de l'entrée :
        # nombre de feature maps, hauteur, largeur
        n_channels, height, width = x.shape

        # Calcul des dimensions de sortie
        output_height = height // self.pool_size
        output_width = width // self.pool_size

        # Création de la sortie
        output = np.zeros(
            (n_channels, output_height, output_width)
        )

        # Parcours de chaque feature map
        for c in range(n_channels):
            for i in range(output_height):
                for j in range(output_width):

                    # Extraction de la zone de pooling
                    region = x[
                        c,
                        i * self.pool_size:(i + 1) * self.pool_size,
                        j * self.pool_size:(j + 1) * self.pool_size
                    ]

                    # Conservation de la valeur maximale
                    output[c, i, j] = np.max(region)

        return output

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


    # =========================
    # Test Conv2D
    # =========================
    print("\n=== Test Conv2D ===")

    x_conv = np.array([
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0]
    ])

    conv = Conv2D(n_filters=2, kernel_size=3)

    output_conv = conv.forward(x_conv)

    print("Entrée :")
    print(x_conv)
    print("Shape entrée :", x_conv.shape)

    print("\nShape kernels :", conv.kernels.shape)

    print("\nSortie :")
    print(output_conv)
    print("Shape sortie :", output_conv.shape)


    # =========================
    # Test MaxPool2D
    # =========================
    print("\n=== Test MaxPool2D ===")

    x_pool = np.array([
        [
            [1, 5, 2, 3],
            [3, 2, 8, 1],
            [4, 2, 3, 7],
            [6, 1, 2, 4]
        ]
    ])

    maxpool = MaxPool2D(pool_size=2)
    output_pool = maxpool.forward(x_pool)

    print("Entrée :")
    print(x_pool)
    print("Shape entrée :", x_pool.shape)

    print("\nSortie :")
    print(output_pool)
    print("Shape sortie :", output_pool.shape)