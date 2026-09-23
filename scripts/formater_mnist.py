import os
import glob
import cv2
import numpy as np

def to_mnist_28x28(gray_cell):
    """
    Prend une case brute en niveaux de gris et la convertit au format strict MNIST 28x28.
    """
    # 1. Binarisation inverse (fond devient noir, encre devient blanche)
    blurred = cv2.GaussianBlur(gray_cell, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Trouver la boîte englobante exacte du tracé
    pts = cv2.findNonZero(thresh)
    if pts is None or len(pts) < 15:
        return None

    x, y, w, h = cv2.boundingRect(pts)
    roi = thresh[y : y + h, x : x + w]

    # 3. Redimensionner à 20 pixels max en gardant les proportions
    if h > w:
        new_h = 20
        new_w = max(1, int(round(w * 20.0 / h)))
    else:
        new_w = 20
        new_h = max(1, int(round(h * 20.0 / w)))

    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 4. Coller dans un canevas 28x28
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

    # 5. Centrage par centre de masse (méthode officielle MNIST)
    M = cv2.moments(canvas)
    if M["m00"] > 0:
        shift_x = int(round(14.0 - (M["m10"] / M["m00"])))
        shift_y = int(round(14.0 - (M["m01"] / M["m00"])))
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        return cv2.warpAffine(canvas, matrix, (28, 28))

    return canvas


def process_folder(input_dir, output_dir):
    print(f"\nTraitement : {input_dir} -> {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for label in range(10):
        src_folder = os.path.join(input_dir, str(label))
        dst_folder = os.path.join(output_dir, str(label))
        os.makedirs(dst_folder, exist_ok=True)

        img_paths = glob.glob(os.path.join(src_folder, "*.png"))
        for img_path in img_paths:
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue

            mnist_img = to_mnist_28x28(gray)
            if mnist_img is not None:
                filename = os.path.basename(img_path)
                cv2.imwrite(os.path.join(dst_folder, filename), mnist_img)
                count += 1

    print(f"-> {count} images converties en 28x28.")


if __name__ == "__main__":
    # Formater les images d'entraînement
    process_folder(
        input_dir="dataset_perso/chiffres_separes",
        output_dir="dataset_perso/chiffres_28x28"
    )

    # Formater aussi le test set pour qu'il soit au format MNIST
    process_folder(
        input_dir="dataset_perso/testset_perso",
        output_dir="dataset_perso/testset_28x28"
    )