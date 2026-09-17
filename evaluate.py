import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def evaluate_model(model, dataloader, device="cpu"):
    """
    Évalue un modèle PyTorch sur un DataLoader donné.
    Retourne :
      - y_true : labels réels (numpy array)
      - y_pred : classes prédites (numpy array)
      - y_probs : probabilités associées à la classe prédite (numpy array)
      - all_images : tenseurs des images d'entrée (torch.Tensor)
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []
    all_probs = []
    all_images = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(confidences.cpu().numpy())
            all_images.append(images.cpu())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    all_images = torch.cat(all_images, dim=0)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n--- Résultats globaux ---")
    print(f"Précision globale (Accuracy) : {acc * 100:.2f} %")
    print("\n--- Rapport détaillé par chiffre (0-9) ---")
    print(classification_report(y_true, y_pred, digits=4))

    return y_true, y_pred, y_probs, all_images


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_path="learning_curves.png"):
    """
    Trace et sauvegarde les courbes de perte (Loss) et d'accuracy (Train vs Validation).
    Permet de repérer le sous-apprentissage ou le surapprentissage.
    """
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Perte (Loss)
    ax1.plot(epochs, train_losses, "b-o", label="Train Loss")
    ax1.plot(epochs, val_losses, "r--s", label="Validation Loss")
    ax1.set_title("Évolution de la fonction de perte (Loss)")
    ax1.set_xlabel("Époques")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Précision (Accuracy)
    ax2.plot(epochs, train_accs, "b-o", label="Train Accuracy")
    ax2.plot(epochs, val_accs, "r--s", label="Validation Accuracy")
    ax2.set_title("Évolution de la précision (Accuracy)")
    ax2.set_xlabel("Époques")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[OK] Courbes d'apprentissage enregistrées : {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """
    Génère et sauvegarde la matrice de confusion 10x10.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(range(10)),
        yticklabels=list(range(10)),
        cbar=True,
    )
    plt.title("Matrice de confusion des chiffres (0 à 9)")
    plt.xlabel("Chiffre prédit")
    plt.ylabel("Chiffre réel")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[OK] Matrice de confusion enregistrée : {save_path}")
    plt.close()


def display_errors(images, y_true, y_pred, y_probs, max_samples=10, save_path="error_analysis.png"):
    """
    Identifie et affiche une grille d'images où le modèle a échoué,
    avec la classe réelle, la prédiction et le score de confiance.
    """
    error_indices = np.where(y_true != y_pred)[0]
    total_errors = len(error_indices)
    print(f"\nNombre total d'erreurs identifiées : {total_errors}")

    if total_errors == 0:
        print("Aucune erreur à afficher.")
        return

    n_display = min(total_errors, max_samples)
    fig, axes = plt.subplots(1, n_display, figsize=(2.5 * n_display, 3))
    if n_display == 1:
        axes = [axes]

    for i in range(n_display):
        idx = error_indices[i]
        img = images[idx].squeeze().numpy()

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(
            f"Vrai: {y_true[idx]}\nPréd: {y_pred[idx]}\n({y_probs[idx]*100:.1f}%)",
            fontsize=10,
            color="red",
        )
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[OK] Échantillon d'erreurs enregistré : {save_path}")
    plt.close()


# if __name__ == "__main__":
#     """
#     Test unitaire autonome avec des données simulées pour vérifier que 
#     tous les tracés et calculs fonctionnent avant l'intégration d'équipe.
#     """
#     print("--- Test autonome du module evaluate.py ---")

#     # 1. Test des courbes d'apprentissage
#     dummy_train_loss = [2.3, 1.5, 0.8, 0.4, 0.2]
#     dummy_val_loss = [2.3, 1.6, 0.9, 0.5, 0.4]
#     dummy_train_acc = [0.20, 0.55, 0.78, 0.89, 0.95]
#     dummy_val_acc = [0.18, 0.50, 0.74, 0.85, 0.90]
#     plot_learning_curves(dummy_train_loss, dummy_val_loss, dummy_train_acc, dummy_val_acc)

#     # 2. Test des métriques et matrice de confusion
#     np.random.seed(42)
#     dummy_true = np.random.randint(0, 10, size=200)
#     dummy_pred = dummy_true.copy()
#     # On force volontairement quelques erreurs
#     error_idx = np.random.choice(200, size=25, replace=False)
#     dummy_pred[error_idx] = np.random.randint(0, 10, size=25)
#     dummy_probs = np.random.uniform(0.60, 0.99, size=200)

#     plot_confusion_matrix(dummy_true, dummy_pred)

#     # 3. Test de la visualisation des erreurs
#     dummy_images = torch.rand(200, 1, 28, 28)
#     display_errors(dummy_images, dummy_true, dummy_pred, dummy_probs, max_samples=8)

#     print("\nTous les tests se sont terminés avec succès !")