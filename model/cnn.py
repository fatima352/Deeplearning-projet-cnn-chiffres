import numpy as np

from model.layers import Conv2D, ReLU, MaxPool2D, Linear, Softmax


class CNN:
    def __init__(self):
        # Extraction des caractéristiques de l'image
        self.conv = Conv2D(n_filters=8, kernel_size=3)
        self.relu = ReLU()
        self.pool = MaxPool2D(pool_size=2)
        self.fc = Linear(n_inputs=8*13*13, n_outputs=10)
        self.softmax = Softmax()

    def forward(self, x):
        # Convolution
        x = self.conv.forward(x)

        # Fonction d'activation
        x = self.relu.forward(x)

        # Réduction des feature maps
        x = self.pool.forward(x)
        self.pre_flatten_shape = x.shape
        x = x.flatten()

        # Classification
        x = self.fc.forward(x)
        x = self.softmax.forward(x)
        return x

    def backward(self, dloss_dprob):
        # 1. On passe d'abord dans la dernière couche (Softmax)
        dx = self.softmax.backward(dloss_dprob)
        
        # 2. Puis la couche linéaire (Fully Connected)
        dx, dw_fc, db_fc = self.fc.backward(dx)
        
        # 3. On redonne la forme 2D (Unflatten)
        dx = dx.reshape(self.pre_flatten_shape)
        
        # 4. On passe dans le MaxPool
        dx = self.pool.backward(dx)
        
        # 5. On passe dans la fonction d'activation ReLU
        dx = self.relu.backward(dx)
        
        # 6. Et enfin la Convolution
        dkernels = self.conv.backward(dx)
        
        return dkernels, dw_fc, db_fc


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