import sys
from pathlib import Path

import numpy as np
import streamlit as st
import cv2
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.cnn import CNN
from scripts.detecteur import detect_digits


def load_trained_model():
    model = CNN()

    saved_model = np.load(PROJECT_ROOT / "modele_chiffres.npz")

    model.conv.kernels = saved_model["conv_kernels"]
    model.fc.weights = saved_model["fc_weights"]
    model.fc.bias = saved_model["fc_bias"]

    return model


def load_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


st.set_page_config(
    page_title="Reconnaissance de chiffres",
    page_icon="🔢",
    layout="centered",
)

st.title("Reconnaissance de chiffres manuscrits")

st.write(
    "Importez une image contenant un chiffre manuscrit."
)

model = load_trained_model()

uploaded_file = st.file_uploader(
    "Choisissez une image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image_rgb = load_uploaded_image(uploaded_file)

    st.image(
        image_rgb,
        caption="Image sélectionnée"
    )

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    digit_tensors, boxes, annotated_image_bgr = detect_digits(image_bgr, debug=False)
    annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

    st.image(
        annotated_image_rgb,
        caption="Chiffres détectés"
    )

    st.write(f"{len(boxes)} chiffre(s) détecté(s).")

    if digit_tensors:
        st.write("Chiffres détectés et extraits de votre image :")
        columns = st.columns(len(digit_tensors))
        for index, (column, digit_tensor) in enumerate(zip(columns, digit_tensors), start=1):
            digit_image = digit_tensor.squeeze()
            column.image(
                digit_image,
                caption=f"#{index}",
                clamp=True
            )
