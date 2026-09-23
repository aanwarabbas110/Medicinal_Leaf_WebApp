
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Medicinal Leaf Identifier",
    page_icon="🌿",
    layout="centered"
)


# ==========================================
# MODEL
# ==========================================

MODEL_PATH = "/content/Medicinal_Leaf_Final_74_40.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# ==========================================
# CLASS NAMES
# ==========================================

class_names = [
    'Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'Astma_weed',
    'Badipala', 'Balloon_Vine', 'Bamboo', 'Beans', 'Betel',
    'Bhrami', 'Bringaraja', 'Caricature', 'Castor', 'Catharanthus',
    'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Coffee',
    'Common rue(naagdalli)', 'Coriender', 'Curry', 'Doddpathre',
    'Drumstick', 'Ekka', 'Eucalyptus', 'Ganigale', 'Ganike',
    'Gasagase', 'Ginger', 'Globe Amarnath', 'Guava', 'Henna',
    'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine',
    'Kambajala', 'Kasambruga', 'Kohlrabi', 'Lantana', 'Lemon',
    'Lemongrass', 'Malabar_Nut', 'Malabar_Spinach', 'Mango',
    'Marigold', 'Mint', 'Neem', 'Nelavembu', 'Nerale', 'Nooni',
    'Onion', 'Padri', 'Palak(Spinach)', 'Papaya', 'Parijatha',
    'Pea', 'Pepper', 'Pomoegranate', 'Pumpkin', 'Raddish',
    'Rose', 'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala',
    'Spinach1', 'Tamarind', 'Taro', 'Tecoma', 'Thumbe', 'Tomato',
    'Tulsi', 'Turmeric', 'ashoka', 'camphor', 'kamakasturi', 'kepala'
]


# ==========================================
# HEADER
# ==========================================

st.title("🌿 Medicinal Leaf Identifier")

st.write(
    "Upload a medicinal leaf image and let the AI "
    "identify the most likely plant."
)

st.info("AI Model Test Accuracy: 74.40%")


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📷 Upload Leaf Image",
    type=["jpg", "jpeg", "png", "webp"]
)


# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf",
        use_container_width=True
    )

    if st.button("🔍 Identify Plant", use_container_width=True):

        with st.spinner("AI is analyzing the leaf..."):

            img = image.resize((224, 224))

            img_array = np.array(img)

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                img_array
            )

            predictions = model.predict(
                img_array,
                verbose=0
            )[0]

            predicted_index = np.argmax(predictions)

            predicted_class = class_names[predicted_index]

            confidence = predictions[predicted_index] * 100


        st.success("🌱 Prediction Complete!")

        st.subheader(
            f"Plant: {predicted_class}"
        )

        st.metric(
            "AI Confidence",
            f"{confidence:.2f}%"
        )


        # ==================================
        # TOP 5
        # ==================================

        st.subheader("🔎 Top 5 Predictions")

        top_indices = np.argsort(predictions)[-5:][::-1]

        for rank, index in enumerate(top_indices, 1):

            plant = class_names[index]

            score = predictions[index] * 100

            st.write(
                f"**{rank}. {plant}** — {score:.2f}%"
            )


# ==========================================
# DISCLAIMER
# ==========================================

st.markdown("---")

st.caption(
    "⚠️ This AI tool is for educational and plant-identification "
    "purposes only. It is not a substitute for professional "
    "medical advice."
)

st.caption("🌿 Medicinal Leaf Identifier | AI Project")
