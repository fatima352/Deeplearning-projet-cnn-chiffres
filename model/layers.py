import numpy as np


class ReLU:
    def __init__(self):
        pass

    # Remplace les valeurs négatives par 0
    def forward(self, x):
        self.input = x #sauvegarde de l'input pour le backward 
        return np.maximum(0, x)

    def backward(self, dx):
        return dx * (self.input > 0)


class Sigmoid:
    def forward(self, x):
        self.output = 1.0 / (1.0 + np.exp(-x))
        return self.output

    def backward(self, dx):
        return dx * self.output * (1.0 - self.output)


class Linear:
    def __init__(self, n_inputs, n_outputs):
        # Initialisation des poids de la couche
        self.weights = np.random.randn(n_outputs, n_inputs)
        # Initialisation des biais à 0
        self.bias = np.zeros(n_outputs)

    def forward(self, x):
        self.input = x #sauvegarde de l'input pour le backward
        return np.matmul(self.weights, x) + self.bias

    def backward(self, dx):
        dw = np.outer(dx, self.input)
        db = dx
        dx_in = np.matmul(self.weights.T, dx)
        return dx_in, dw, db


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
        self.input = x
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

    def backward(self, dx):
        dkernels = np.zeros_like(self.kernels)
        for f in range(self.n_filters):
            for i in range(dx.shape[1]):
                for j in range(dx.shape[2]):
                    region = self.input[i:i+self.kernel_size, j:j+self.kernel_size]
                    dkernels[f] += dx[f, i, j] * region
        return dkernels

class MaxPool2D:
    def __init__(self, pool_size=2):
        # Taille de la fenêtre de pooling
        self.pool_size = pool_size

    def forward(self, x):
        self.input = x #sauvegarde input pour le backward

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

    def backward(self, dx):
        n_channels, height, width = self.input.shape
        dx_in = np.zeros_like(self.input)
        for c in range(n_channels):
            for i in range(dx.shape[1]):
                for j in range(dx.shape[2]):
                    region = self.input[c, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size]
                    max_pos = np.unravel_index(np.argmax(region), region.shape)
                    dx_in[c, i*self.pool_size + max_pos[0], j*self.pool_size + max_pos[1]] = dx[c, i, j]
        return dx_in

class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        # Le np.max sécurise le calcul pour ne jamais avoir d'overflow
        exp_x = np.exp(x - np.max(x))
        self.output = exp_x / np.sum(exp_x)
        return self.output

    def backward(self, dloss_dprob):
        # La soustraction (prediction - y) est déjà faite dans train.py.
        # Cette couche ne fait donc que relayer le gradient mathématique.
        return dloss_dprob