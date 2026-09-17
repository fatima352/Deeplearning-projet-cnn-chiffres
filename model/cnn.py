import numpy as np

from model.layers import Conv2D, ReLU, MaxPool2D, Linear


class CNN:
    def __init__(self):
        # Extraction des caractéristiques de l'image
        self.conv = Conv2D(n_filters=8, kernel_size=3)
        self.relu = ReLU()
        self.pool = MaxPool2D(pool_size=2)

        # Classification en 10 classes : chiffres de 0 à 9
        self.fc = Linear(
            n_inputs=8 * 13 * 13,
            n_outputs=10
        )

    def forward(self, x):
        # Convolution
        x = self.conv.forward(x)

        # Fonction d'activation
        x = self.relu.forward(x)

        # Réduction des feature maps
        x = self.pool.forward(x)

        # Transformation en vecteur
        x = x.flatten()

        # Classification
        x = self.fc.forward(x)

        return x


if __name__ == "__main__":
    import numpy as np

    print("=== Test CNN.forward() ===")

    # Création d'une fausse image 28x28
    x = np.random.rand(28, 28)

    # Création du modèle
    model = CNN()

    # Passage de l'image dans tout le CNN
    output = model.forward(x)

    print("Shape entrée :", x.shape)
    print("Shape sortie :", output.shape)

    print("\nScores de sortie :")
    print(output)