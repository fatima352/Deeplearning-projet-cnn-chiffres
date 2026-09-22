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


def prepare_digit_for_model(digit_tensor):
    digit_image = digit_tensor.squeeze()
    return (digit_image - 0.1307) / 0.3081


def predict_digit(model, digit_tensor):
    model_input = prepare_digit_for_model(digit_tensor)
    probabilities = model.forward(model_input)
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])
    return predicted_digit, confidence, probabilities


# --------------------------------------------------
# Configuration de la page
# --------------------------------------------------

st.set_page_config(
    page_title="Reconnaissance de chiffres",
    page_icon="🔢",
    layout="centered",
)


# --------------------------------------------------
# Style de l'interface
# --------------------------------------------------

st.markdown(
    """
    <style>

    /* Fond général */
    .stApp {
        background: linear-gradient(
            135deg,
            #eef2ff 0%,
            #f5f7ff 45%,
            #eef6ff 100%
        );
    }

    /* Largeur et espacement du contenu */
    .block-container {
        max-width: 950px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    /* Titres */
    h1 {
        color: #1e293b;
        font-weight: 700;
        text-align: center;
        letter-spacing: -1px;
    }

    h2, h3, h4 {
        color: #1e293b;
    }

    /* Texte général */
    p {
        color: #475569;
    }

    /* Zone d'introduction */
    .intro {
        text-align: center;
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Titres des différentes étapes */
    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        color: #1e293b;
        margin-top: 2rem;
        margin-bottom: 0.3rem;
    }

    .section-description {
        color: #64748b;
        margin-bottom: 1.2rem;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.85);
        border: 1px solid #dbe3f0;
        border-radius: 14px;
        padding: 1.2rem;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 4px 14px rgba(30, 41, 59, 0.06);
    }

    [data-testid="stMetricValue"] {
        color: #3730a3;
        font-weight: 700;
    }

    /* Messages */
    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }

    /* Séparateurs */
    hr {
        border-color: #dbe3f0;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# En-tête
# --------------------------------------------------

st.title("Reconnaissance de chiffres manuscrits")

st.markdown(
    """
    <div class="intro">
        Importez une image contenant un ou plusieurs chiffres manuscrits.
        L'application détectera automatiquement les chiffres présents
        et affichera le résultat de la reconnaissance.
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Chargement du modèle
# --------------------------------------------------

model = load_trained_model()


# --------------------------------------------------
# Import de l'image
# --------------------------------------------------

st.markdown(
    '<div class="section-title">1. Importer une image</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Sélectionnez une image au format PNG, JPG ou JPEG.'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choisissez une image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    image_rgb = load_uploaded_image(uploaded_file)

    st.image(
        image_rgb,
        caption="Image importée",
        use_container_width=True
    )

    st.divider()


    # --------------------------------------------------
    # Détection
    # --------------------------------------------------

    st.markdown(
        '<div class="section-title">2. Détection des chiffres</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        "Les chiffres repérés dans l'image sont encadrés automatiquement."
        '</div>',
        unsafe_allow_html=True
    )

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    digit_tensors, boxes, annotated_image_bgr = detect_digits(image_bgr, debug=False)
    annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

    st.image(
        annotated_image_rgb,
        caption="Résultat de la détection",
        use_container_width=True
    )

    if len(boxes) == 1:
        st.success("1 chiffre a été détecté dans l'image.")
    else:
        st.success(
            f"{len(boxes)} chiffres ont été détectés dans l'image."
        )


    # --------------------------------------------------
    # Chiffres extraits
    # --------------------------------------------------

    if digit_tensors:

        st.divider()

        st.markdown(
            '<div class="section-title">3. Chiffres identifiés</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">'
            "Voici les chiffres isolés avant leur reconnaissance par le modèle."
            '</div>',
            unsafe_allow_html=True
        )

        columns = st.columns(len(digit_tensors))
        for index, (column, digit_tensor) in enumerate(zip(columns, digit_tensors), start=1):
            digit_image = digit_tensor.squeeze()
            predicted_digit, confidence, probabilities = predict_digit(model, digit_tensor)

            column.image(
                digit_image,
                caption=f"Image {index}",
                clamp=True
            )


        # --------------------------------------------------
        # Résultats
        # --------------------------------------------------

        for index, digit_tensor in enumerate(digit_tensors, start=1):
            predicted_digit, confidence, probabilities = predict_digit(model, digit_tensor)

            digit_image = digit_tensor.squeeze()

            st.markdown(f"#### Image {index}")

            image_col, result_col, confidence_col = st.columns(
                [1, 2, 2],
                vertical_alignment="center"
            )

            with image_col:
                st.image(
                    digit_image,
                    width=100,
                    clamp=True
                )

            with result_col:
                st.metric(
                    label="Chiffre reconnu",
                    value=str(predicted_digit)
                )

            with confidence_col:
                st.metric(
                    label="Niveau de confiance",
                    value=f"{confidence:.1%}"
                )

            with st.expander("Afficher le détail des probabilités"):
                st.bar_chart({str(digit): float(probability) for digit, probability in enumerate(probabilities)})

            if index < len(digit_tensors):
                st.markdown("---")

                
    else:
        st.warning(
            "Aucun chiffre n'a été détecté. "
            "Essayez avec une image plus nette ou mieux cadrée."
        )