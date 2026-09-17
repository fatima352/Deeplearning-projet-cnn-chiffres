import os
import cv2
import numpy as np

INPUT_IMAGE = "grilles/Rayan.jpg"
OUTPUT_DIR = "chiffres_separes"
AUTHOR = "rayan"
NB_ROWS = 10
NB_COLS = 10


os.makedirs(OUTPUT_DIR, exist_ok=True)
img = cv2.imread(INPUT_IMAGE)
img = cv2.copyMakeBorder(
    img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255]
)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Binarisation inversée : les traits deviennent blancs
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Détection des lignes horizontales et verticales
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

lines_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
lines_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

# Fusion pour reconstruire la grille vide
table_grid = cv2.add(lines_h, lines_v)

# Trouver les contours de chaque case individuelle
contours, _ = cv2.findContours(
    table_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)

boxes = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    # Filtrer les bordures globales ou les micro-bruits
    if (
        w > img.shape[1] * 0.05
        and h > img.shape[0] * 0.05
        and w < img.shape[1] * 0.5
    ):
        boxes.append((x, y, w, h))

# Trier les cases de haut en bas, puis de gauche à droite
# (regroupement par ligne avec une tolérance en Y)
boxes = sorted(boxes, key=lambda b: b[1])
rows = []
for box in boxes:
    if not rows or abs(box[1] - rows[-1][0][1]) > (box[3] * 0.5):
        rows.append([box])
    else:
        rows[-1].append(box)

# Vérification du nombre de cases détectées
total_detected = sum(len(r) for r in rows)
print(f"Lignes détectées : {len(rows)}, Cases totales : {total_detected}")

count = 0
for row_idx, row_boxes in enumerate(rows[:10]):
    row_boxes = sorted(row_boxes, key=lambda b: b[0])  # tri de gauche à droite
    label = str(row_idx)
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

    for col_idx, (x, y, w, h) in enumerate(row_boxes[:10]):
        # Rognage intérieur de la case
        pad_y = int(h * 0.08)
        pad_x = int(w * 0.08)
        cell = img[y + pad_y : y + h - pad_y, x + pad_x : x + w - pad_x]

        filename = f"{label}_{AUTHOR}_r{row_idx}_c{col_idx}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, label, filename), cell)
        count += 1

print(f"Extraction terminée : {count} cases enregistrées.")