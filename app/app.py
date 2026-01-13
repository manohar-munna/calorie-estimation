import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

MODEL_PATH = "models/food_model.h5"
CALORIES_PATH = "calories.csv"

st.title("🍽️ Food & Beverage Calories Estimation")
st.write("Upload an image of Biryani / Coke / Pizza and get calories ✅")

model = tf.keras.models.load_model(MODEL_PATH)

cal_df = pd.read_csv(CALORIES_PATH)
cal_map = dict(zip(cal_df["class"], cal_df["calories"]))

index_to_class = {0: "biryani", 1: "coke", 2: "pizza"}

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0]
    pred_index = int(np.argmax(pred))
    confidence = float(pred[pred_index]) * 100

    pred_class = index_to_class[pred_index]
    calories = cal_map.get(pred_class, "Not found")

    st.success(f"✅ Prediction: {pred_class}")
    st.info(f"🔥 Calories: {calories} kcal")
    st.write(f"Confidence: {confidence:.2f}%")
