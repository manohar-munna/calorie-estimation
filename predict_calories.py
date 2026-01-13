import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image

MODEL_PATH = "models/food_model.h5"
CALORIES_PATH = "calories.csv"

model = tf.keras.models.load_model(MODEL_PATH)

cal_df = pd.read_csv(CALORIES_PATH)
cal_map = dict(zip(cal_df["class"], cal_df["calories"]))

index_to_class = {0: "biryani", 1: "coke", 2: "pizza"}

img_path = input("Enter image path: ").strip()

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

pred = model.predict(img_array)[0]
pred_index = int(np.argmax(pred))
confidence = float(pred[pred_index]) * 100

pred_class = index_to_class[pred_index]
calories = cal_map.get(pred_class, "Not found")

print("\n✅ Prediction Result")
print("Food:", pred_class)
print("Confidence:", round(confidence, 2), "%")
print("Estimated Calories:", calories, "kcal")
