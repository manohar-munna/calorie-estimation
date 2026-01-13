from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "dataset/train"

train_gen = ImageDataGenerator(rescale=1.0/255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical",
    shuffle=False
)

print("✅ Class mapping:", train_data.class_indices)
