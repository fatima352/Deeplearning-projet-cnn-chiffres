import streamlit as st


st.set_page_config(
    page_title="Reconnaissance de chiffres",
    page_icon="🔢",
    layout="centered",
)

st.title("Reconnaissance de chiffres manuscrits")

st.write(
    "Importez une image contenant un chiffre manuscrit."
)

uploaded_file = st.file_uploader(
    "Choisissez une image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Image sélectionnée"
    )