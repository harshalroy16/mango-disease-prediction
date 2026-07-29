import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

# -----------------------------
# Dataset Settings
# -----------------------------
DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# -----------------------------
# Load Dataset
# -----------------------------
train_dataset = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# -----------------------------
# Class Names
# -----------------------------
class_names = train_dataset.class_names
num_classes = len(class_names)

print("Disease Classes:", class_names)

# -----------------------------
# Build CNN Model
# -----------------------------
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Save the trained model
model.save("models/mango_disease_model.keras")

print("\n✅ Model trained successfully!")
print("✅ Model saved in models/mango_disease_model.keras")