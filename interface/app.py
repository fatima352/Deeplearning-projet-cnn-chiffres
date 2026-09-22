import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.cnn import CNN

def load_trained_model():
    model = CNN()

    saved_model = np.load(PROJECT_ROOT / "modele_chiffres.npz")

    model.conv.kernels = saved_model["conv_kernels"]
    model.fc.weights = saved_model["fc_weights"]
    model.fc.bias = saved_model["fc_bias"]

    return model


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
    st.image(
        uploaded_file,
        caption="Image sélectionnée"
    )