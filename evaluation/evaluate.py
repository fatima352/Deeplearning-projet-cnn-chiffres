import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import de votre CNN from scratch (P2)
from model.cnn import CNN


def evaluate_model_numpy(model, dataloader):
    """
    Évalue le CNN NumPy sur un DataLoader PyTorch.
    Retourne : y_true, y_pred, y_probs, images_list, avg_loss
    """
    all_preds = []
    all_targets = []
    all_probs = []
    all_images = []
    total_loss = 0.0
    total_samples = 0

    for batch_images, batch_labels in dataloader:
        for i in range(len(batch_images)):
            img_tensor = batch_images[i]
            x = img_tensor.squeeze().numpy()  # Format 28x28
            label = batch_labels[i].item()

            # Encodage One-Hot
            y = np.zeros(10)
            y[label] = 1.0

            # Inférence NumPy
            probs = model.forward(x)  # Vecteur de probabilités (Softmax déjà appliqué en sortie de P2)

            # Calcul de la perte Cross-Entropy
            loss = -np.sum(y * np.log(probs + 1e-9))
            total_loss += loss

            pred_label = int(np.argmax(probs))
            confidence = float(np.max(probs))

            all_preds.append(pred_label)
            all_targets.append(label)
            all_probs.append(confidence)
            all_images.append(img_tensor.squeeze().numpy())
            total_samples += 1

    avg_loss = total_loss / total_samples
    return (
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_images),
        avg_loss,
    )


def plot_confusion_matrix(y_true, y_pred, save_path="report/confusion_matrix.png"):
    """Génère et sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(range(10)),
        yticklabels=list(range(10)),
    )
    plt.title("Matrice de confusion des chiffres (0 à 9)")
    plt.xlabel("Chiffre prédit")
    plt.ylabel("Chiffre réel")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"[OK] Matrice de confusion enregistrée : {save_path}")
    plt.close()


def display_errors(images, y_true, y_pred, y_probs, max_samples=8, save_path="report/error_analysis.png"):
    """Visualise les erreurs de classification avec niveau de confiance."""
    error_indices = np.where(y_true != y_pred)[0]
    total_errors = len(error_indices)
    print(f"Nombre total d'erreurs identifiées : {total_errors} / {len(y_true)}")

    if total_errors == 0:
        print("Aucune erreur détectée sur ce jeu de test.")
        return

    n_display = min(total_errors, max_samples)
    fig, axes = plt.subplots(1, n_display, figsize=(2.5 * n_display, 3))
    if n_display == 1:
        axes = [axes]

    for i in range(n_display):
        idx = error_indices[i]
        axes[i].imshow(images[idx], cmap="gray")
        axes[i].set_title(
            f"Vrai: {y_true[idx]}\nPréd: {y_pred[idx]}\n({y_probs[idx]*100:.1f} %)",
            fontsize=10,
            color="red",
        )
        axes[i].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"[OK] Planche d'erreurs enregistrée : {save_path}")
    plt.close()


def plot_learning_curves(history, save_path="report/learning_curves.png"):
    """Génère les courbes d'apprentissage à partir d'un dictionnaire d'historique."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    if "val_loss" in history and len(history["val_loss"]) > 0:
        ax1.plot(epochs, history["val_loss"], "r--s", label="Validation Loss")
    ax1.set_title("Évolution de la Perte (Loss)")
    ax1.set_xlabel("Époques")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc")
    if "val_acc" in history and len(history["val_acc"]) > 0:
        ax2.plot(epochs, history["val_acc"], "r--s", label="Validation Acc")
    ax2.set_title("Évolution de la Précision (Accuracy)")
    ax2.set_xlabel("Époques")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"[OK] Courbes d'entraînement enregistrées : {save_path}")
    plt.close()


def load_model(weights_path):
    """Initialise le CNN et charge les poids sauvegardés."""
    model = CNN()
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Poids introuvables à l'emplacement : {weights_path}")
    data = np.load(weights_path)
    model.conv.kernels = data["conv_kernels"]
    model.fc.weights = data["fc_weights"]
    model.fc.bias = data["fc_bias"]
    return model


if __name__ == "__main__":
    # 1. Pipeline de normalisation (doit être STRICTEMENT identique à celui de P1 et P4)
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_dir = "./dataset_perso/testset_28x28"
    weights_path = "modele_chiffres.npz"

    if not os.path.exists(test_dir):
        print(f"Erreur : Dossier de test introuvable : {test_dir}")
        sys.exit(1)

    test_loader = DataLoader(
        datasets.ImageFolder(root=test_dir, transform=test_transform),
        batch_size=32,
        shuffle=False,
    )

    print(f">> Chargement du modèle depuis {weights_path}...")
    model = load_model(weights_path)

    print(">> Évaluation en cours...")
    y_true, y_pred, y_probs, images, avg_loss = evaluate_model_numpy(model, test_loader)

    # Métriques
    acc = accuracy_score(y_true, y_pred) * 100
    print("\n" + "=" * 40)
    print(f"Loss moyenne sur le test : {avg_loss:.4f}")
    print(f"Précision globale (Accuracy) : {acc:.2f} %")
    print("=" * 40 + "\n")

    print("--- Rapport de classification ---")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    # Graphiques pour le dossier report/
    plot_confusion_matrix(y_true, y_pred, save_path="report/confusion_matrix.png")
    display_errors(images, y_true, y_pred, y_probs, max_samples=8, save_path="report/error_analysis.png")