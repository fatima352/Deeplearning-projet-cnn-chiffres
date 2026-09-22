import os
import sys
import cv2
import numpy as np


def extract_single_digit(roi_thresh):
    """Normalise la région du chiffre en tenseur (1, 28, 28) conforme MNIST."""
    pts = cv2.findNonZero(roi_thresh)
    if pts is None:
        return np.zeros((1, 28, 28), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(pts)
    tight = roi_thresh[y : y + h, x : x + w]

    # Redimensionnement 20 px max
    if h > w:
        new_h = 20
        new_w = max(1, int(round(w * 20.0 / h)))
    else:
        new_w = 20
        new_h = max(1, int(round(h * 20.0 / w)))

    resized = cv2.resize(tight, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

    # Centrage barycentre
    M = cv2.moments(canvas)
    if M["m00"] > 0:
        shift_x = int(round(14.0 - (M["m10"] / M["m00"])))
        shift_y = int(round(14.0 - (M["m01"] / M["m00"])))
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, matrix, (28, 28))

    tensor = (canvas.astype(np.float32) / 255.0)[np.newaxis, :, :]
    return tensor


def detect_digits(image_input, debug=True):
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Image introuvable : {image_input}")
    else:
        img = image_input.copy()

    H_total, W_total = img.shape[:2]
    total_pixels = H_total * W_total
    annotated_img = img.copy()

    # 1. Niveaux de gris
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 2. Détection automatique du fond (fond clair vs fond sombre)
    # On regarde si la moyenne des pixels est plutôt claire (>127) ou sombre
    mean_val = np.mean(gray)
    is_white_background = mean_val > 127

    if is_white_background:
        # Papier blanc, encre noire -> on inverse
        blurred = cv2.GaussianBlur(gray, (5, 5), 0) if total_pixels > 2000 else gray
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        # Déjà fond noir et tracé blanc (ex: MNIST)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # 3. Recherche des contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if debug:
        print(f"[Debug] Taille image : {W_total}x{H_total}")
        print(f"[Debug] Fond détecté : {'Blanc (inversé)' if is_white_background else 'Noir'}")
        print(f"[Debug] Contours bruts trouvés : {len(contours)}")

    # 4. Seuils dynamiques selon la taille de l'image
    # Pour une petite image (ex 28x28), on accepte dès 15 pixels. Pour une photo, min 0.05% de l'image
    min_area = max(15, int(total_pixels * 0.0005))
    max_area = int(total_pixels * 0.95)

    detected_boxes = []

    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h > 0 else 0

        # Vérification des filtres
        if area < min_area:
            if debug and total_pixels < 2000:
                print(f"   Contour #{idx} rejeté : surface trop petite ({area} < {min_area})")
            continue
        if area > max_area:
            continue
        # Accepte les chiffres très fins (comme le 1) jusqu'à aspect_ratio 0.05
        if aspect_ratio > 5.0 or aspect_ratio < 0.05:
            continue

        detected_boxes.append((x, y, w, h))

    # Tri de gauche à droite
    detected_boxes = sorted(detected_boxes, key=lambda b: b[0])

    digits_tensors = []
    final_boxes = []

    for i, (x, y, w, h) in enumerate(detected_boxes):
        pad = max(2, int(min(H_total, W_total) * 0.02))
        y1 = max(0, y - pad)
        y2 = min(H_total, y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(W_total, x + w + pad)

        roi = thresh[y1:y2, x1:x2]
        tensor_28 = extract_single_digit(roi)

        digits_tensors.append(tensor_28)
        final_boxes.append((x, y, w, h))

        # Dessin du rectangle vert
        thickness = max(1, int(min(H_total, W_total) / 150))
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), thickness)
        cv2.putText(
            annotated_img,
            f"#{i+1}",
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 if total_pixels < 5000 else 1.0,
            (0, 255, 0),
            thickness,
        )

    return digits_tensors, final_boxes, annotated_img


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python3 detecteur.py <chemin_image>")
        sys.exit(0)

    test_path = sys.argv[1]
    tensors, boxes, marked = detect_digits(test_path, debug=True)

    print(f"\n-> Résultat : {len(boxes)} chiffre(s) détecté(s).")
    for idx, (t, b) in enumerate(zip(tensors, boxes)):
        print(f"   Chiffre #{idx+1} : Bounding Box (x={b[0]}, y={b[1]}, w={b[2]}, h={b[3]}) | Tenseur shape = {t.shape}")

    output_preview = "detection_result.png"
    cv2.imwrite(output_preview, marked)
    print(f"Image avec encadrements sauvegardée sous : {output_preview}")