import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import pandas as pd
import io

# -------------- Fungsi utilitas -------------- #
@st.cache_resource
def load_tflite_model(path: str):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image_file, size: int = 224):
    img = Image.open(image_file).convert("RGB")
    img_resized = img.resize((size, size))
    return img, img_resized

def predict_image(interpreter, img_resized):
    # Preprocessing: ubah jadi array
    arr = img_to_array(img_resized)
    arr = np.expand_dims(arr, axis=0).astype(np.float32) / 255.0

    # Ambil input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input data
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()

    # Ambil hasil prediksi
    preds = interpreter.get_tensor(output_details[0]['index'])
    return preds[0]

# Label kelas
LABELS = ['Normal', 'Pneumonia', 'Tuberculosis']

# Path model TFLite yang digunakan
MODEL_PATH = "./model.tflite"   # ganti sesuai nama file kamu

# Inisialisasi pilihan halaman
if 'choice' not in st.session_state:
    st.session_state.choice = 'Preprocessing Gambar'

# ---------------- Ganti Halaman ---------------- #
def set_choice_pre():   st.session_state.choice = 'Preprocessing Gambar'
def set_choice_pred():  st.session_state.choice = 'Klasifikasi Penyakit Paru-paru'

# ---------------- MAIN ---------------- #
def main():
    st.set_page_config(page_title="Klasifikasi Paru-Paru", layout="wide")
    st.title("Aplikasi Klasifikasi Penyakit Paru-Paru")
    st.write("Aplikasi ini mengklasifikasikan citra Rontgen paru menjadi "
             "**Normal**, **Pneumonia**, atau **Tuberculosis**.")

    # ---------- Styling tombol kotak panjang ---------- #
    st.markdown("""
        <style>
        button[kind="secondary"] {
            height: 50px !important;
            font-size: 18px !important;
            border-radius: 10px !important;
            margin-bottom: 10px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # -------- Menu kotak di halaman utama -------- #
    st.markdown("### Pilih Menu")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Preprocessing Gambar", use_container_width=True):
            set_choice_pre()
    with col2:
        if st.button("Klasifikasi Penyakit Paru-paru", use_container_width=True):
            set_choice_pred()

    choice = st.session_state.choice

    # -------- PREPROCESSING -------- #
    if choice == "Preprocessing Gambar":
        st.subheader("Preprocessing Gambar Rontgen")

        uploaded_file = st.file_uploader(
            "Unggah gambar rontgen", 
            type=["jpg", "jpeg", "png"], 
            key="preprocess_uploader"
        )

        if uploaded_file:
            original_img, resized_img = preprocess_image(uploaded_file, 224)

            col1, col2 = st.columns(2)

            with col1:
                left_space, center_col, right_space = st.columns([1, 2, 1])
                with center_col:
                    st.image(
                        original_img, 
                        caption=f"Gambar Sebelum Resize ({original_img.width}×{original_img.height})", 
                        width=300
                    )

            with col2:
                left_space, center_col, right_space = st.columns([1, 2, 1])
                with center_col:
                    st.image(
                        resized_img, 
                        caption="Gambar Setelah Resize (224×224)", 
                        width=300
                    )

            # Simpan hasil preprocessing ke session
            buf = io.BytesIO()
            resized_img.save(buf, format="PNG")
            st.session_state.preprocessed_image = buf.getvalue()

    # -------- KLASIFIKASI -------- #
    elif choice == "Klasifikasi Penyakit Paru-paru":
        st.subheader("Klasifikasi Penyakit Paru-paru")

        if 'preprocessed_image' in st.session_state:
            col1, col2 = st.columns([1, 1.2])

            with col1:
                left_space, center_col, right_space = st.columns([1, 2, 1])
                with center_col:
                    img_bytes = st.session_state.preprocessed_image
                    img = Image.open(io.BytesIO(img_bytes))
                    st.image(img, caption="Gambar Rontgen (224×224)", width=300)

            with col2:
                interpreter = load_tflite_model(MODEL_PATH)
                preds = predict_image(interpreter, img)
                pred_label = LABELS[np.argmax(preds)]

                st.markdown("### Hasil Klasifikasi")
                st.success(f"**{pred_label}**")

                prob_df = pd.DataFrame({
                    "Kelas": LABELS,
                    "Probabilitas": [f"{p * 100:.2f}%" for p in preds]
                })

                st.markdown("### Probabilitas Tiap Kelas")
                st.table(prob_df)

        else:
            st.warning("Silakan unggah dan proses gambar terlebih dahulu di menu 'Preprocessing Gambar'.")

if __name__ == "__main__":
    main()
