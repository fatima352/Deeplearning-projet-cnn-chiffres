import os
import glob
import random
import cv2
import numpy as np

# ==========================================
# CONFIGURATION DES DOSSIERS
# ==========================================
# Adapte le chemin si ton script est dans "scripts/" ou à la racine
INPUT_DIR = "dataset_perso/chiffres_28x28"
OUTPUT_DIR = "dataset_perso/dataset_augmente"

# 1 image source -> 1 originale + 9 variantes = 10 images au total
MULTIPLIER = 10 


def augment_digit(img):
    """
    Applique des transformations géométriques et morphologiques douces
    adaptées aux chiffres MNIST 28x28.
    """
    h, w = img.shape

    # 1. Rotation aléatoire douce (-12° à +12°)
    # (Attention : ne pas trop tourner pour ne pas confondre un 6 et un 9)
    angle = random.uniform(-12, 12)
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    img_trans = cv2.warpAffine(
        img, rot_mat, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    # 2. Translation aléatoire (-2 à +2 pixels en X et Y)
    tx = random.randint(-2, 2)
    ty = random.randint(-2, 2)
    trans_mat = np.float32([[1, 0, tx], [0, 1, ty]])
    img_trans = cv2.warpAffine(
        img_trans, trans_mat, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    # 3. Zoom / dézoom léger (échelle entre 0.90 et 1.10)
    scale = random.uniform(0.90, 1.10)
    scaled_w = max(1, int(round(w * scale)))
    scaled_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img_trans, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    # Replacer dans un canevas 28x28 centré
    canvas = np.zeros((28, 28), dtype=np.uint8)
    if scale >= 1.0:
        # Si zoom : on découpe le centre
        y_start = (scaled_h - 28) // 2
        x_start = (scaled_w - 28) // 2
        canvas = resized[y_start : y_start + 28, x_start : x_start + 28]
    else:
        # Si dézoom : on colle au centre
        y_offset = (28 - scaled_h) // 2
        x_offset = (28 - scaled_w) // 2
        canvas[y_offset : y_offset + scaled_h, x_offset : x_offset + scaled_w] = resized

    # 4. Variation d'épaisseur de trait (simulation stylo bille / feutre)
    choice = random.choice(["dilate", "erode", "none", "none"])
    kernel = np.ones((2, 2), np.uint8)

    if choice == "dilate":
        canvas = cv2.dilate(canvas, kernel, iterations=1)
    elif choice == "erode":
        # Vérification qu'il reste de la matière avant d'éroder
        eroded = cv2.erode(canvas, kernel, iterations=1)
        if cv2.countNonZero(eroded) > 20:
            canvas = eroded

    return canvas


def main():
    global INPUT_DIR, OUTPUT_DIR

    # Détection automatique du chemin selon l'endroit où le terminal est ouvert
    if not os.path.exists(INPUT_DIR):
        alt_paths = [
            "../dataset_perso/chiffres_28x28",
            "chiffres_28x28",
            "dataset_perso/chiffres_28x28",
        ]
        found = False
        for p in alt_paths:
            if os.path.exists(p):
                INPUT_DIR = p
                OUTPUT_DIR = os.path.join(os.path.dirname(p), "dataset_augmente")
                found = True
                break

        if not found:
            raise FileNotFoundError(f"Dossier source introuvable. Emplacement testé : {INPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_orig = 0
    total_generated = 0

    print("--- Démarrage de l'augmentation des données ---")
    print(f"Source : {INPUT_DIR}")
    print(f"Cible  : {OUTPUT_DIR}\n")

    for label in range(10):
        src_folder = os.path.join(INPUT_DIR, str(label))
        dst_folder = os.path.join(OUTPUT_DIR, str(label))
        os.makedirs(dst_folder, exist_ok=True)

        img_paths = sorted(glob.glob(os.path.join(src_folder, "*.png")))
        count_class = 0

        for img_path in img_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            total_orig += 1

            # 1. Image originale
            cv2.imwrite(os.path.join(dst_folder, f"{base_name}_orig.png"), img)
            count_class += 1
            total_generated += 1

            # 2. Variantes
            for i in range(MULTIPLIER - 1):
                aug = augment_digit(img)
                aug_name = f"{base_name}_aug_{i}.png"
                cv2.imwrite(os.path.join(dst_folder, aug_name), aug)
                count_class += 1
                total_generated += 1

        print(f"Classe [{label}] : {len(img_paths)} images -> {count_class} générées")

    print("\n------------------------------------------------")
    print("Terminé avec succès !")
    print(f"Images sources lues     : {total_orig}")
    print(f"Total images augmentées : {total_generated}")
    print(f"Dossier final           : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()