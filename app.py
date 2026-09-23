import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Medicinal Leaf Identifier",
    page_icon="🌿",
    layout="centered"
)


# ==========================================
# PLANT CLASSES
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
# MODEL
# ==========================================


MODEL_PATH = "Medicinal_Leaf_Deployment.keras"


@st.cache_resource
def load_model():

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomContrast(0.10),
    ])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=(224, 224, 3))

    x = data_augmentation(inputs)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.0001)
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.30)(x)

    outputs = tf.keras.layers.Dense(
        80,
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.load_weights("Medicinal_Leaf_74_40.weights.h5")

    return model


# ==========================================
# HEADER
# ==========================================

st.title("🌿 Medicinal Leaf Identifier")

st.write(
    "Upload a medicinal plant leaf image and the AI model "
    "will predict the most likely plant."
)

st.info("Model accuracy on test dataset: 74.40%")


# ==========================================
# CHECK MODEL
# ==========================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "Model file not found. Please place "
        "Medicinal_Leaf_Final_74_40.keras in the same folder as app.py."
    )

    st.stop()


model = load_model()


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📷 Upload a leaf image",
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


        # ==================================
        # RESULT
        # ==================================

        st.success("🌿 Prediction Complete!")

        st.subheader(
            f"🌱 Plant: {predicted_class}"
        )

        st.metric(
            "AI Confidence",
            f"{confidence:.2f}%"
        )


        # ==================================
        # TOP 5 PREDICTIONS
        # ==================================

        st.subheader("🔎 Top Predictions")

        top_indices = np.argsort(predictions)[-5:][::-1]

        for i, index in enumerate(top_indices):

            plant = class_names[index]

            score = predictions[index] * 100

            st.write(
                f"**{i + 1}. {plant}** — {score:.2f}%"
            )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "⚠️ This AI tool is for educational and identification purposes only. "
    "It is not a substitute for professional medical advice."
)

st.caption("🌿 Medicinal Leaf Identifier | AI Project")