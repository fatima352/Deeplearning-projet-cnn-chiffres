import os
import sys
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import de ton architecture NumPy
from model.cnn import CNN

def evaluate_model_numpy(model, dataloader):
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_images, batch_labels in dataloader:
        batch_loss = 0.0

        for i in range(len(batch_images)):
            x = batch_images[i].squeeze().numpy()
            label_int = batch_labels[i].item()

            # Encodage One-Hot
            y = np.zeros(10)
            y[label_int] = 1.0

            # Forward pass
            prediction = model.forward(x)

            # Calcul de la Loss (Cross-Entropy)
            loss = -np.sum(y * np.log(prediction + 1e-9))
            batch_loss += loss

            # Vérification de la précision
            predicted_label = np.argmax(prediction)
            if predicted_label == label_int:
                correct += 1
            total += 1

        running_loss += batch_loss / len(batch_images)

    accuracy = 100 * correct / total
    return running_loss / len(dataloader), accuracy


if __name__ == "__main__":
    # 1. Préparation des données de test (sans Data Augmentation ni inversion)
    perso_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("Chargement du jeu de test...")
    test_data = datasets.ImageFolder(
        root='./dataset_perso/testset_28x28',
        transform=perso_test_transform
    )
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # 2. Initialisation et chargement des poids finaux
    print("Chargement des poids depuis 'modele_chiffres.npz'...")
    model = CNN()
    
    try:
        sauvegarde = np.load("modele_chiffres.npz")
        model.conv.kernels = sauvegarde['conv_kernels']
        model.fc.weights = sauvegarde['fc_weights']
        model.fc.bias = sauvegarde['fc_bias']
    except FileNotFoundError:
        print("Erreur : Le fichier 'modele_chiffres.npz' est introuvable. Vérifie son emplacement.")
        sys.exit(1)

    # 3. Évaluation
    print(f"Évaluation en cours sur {len(test_data)} images...\n")
    test_loss, test_acc = evaluate_model_numpy(model, test_loader)

    print("-" * 30)
    print("🏆 RÉSULTAT FINAL DU MODÈLE")
    print("-" * 30)
    print(f"Test Loss : {test_loss:.4f}")
    print(f"Accuracy  : {test_acc:.2f} %")
    print("-" * 30)