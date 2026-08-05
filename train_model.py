import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize
# -----------------------------
# Dataset Settings
# -----------------------------
DATASET_DIR = r"C:\Users\HP\Desktop\temp\dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

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

class_names = train_dataset.class_names
num_classes = len(class_names)

print("\nDisease Classes:")
print(class_names)
print("\nClass Index Mapping:")
for i, name in enumerate(class_names):
    print(f"{i} -> {name}")
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = (
    train_dataset
    .cache()
    .shuffle(1000)
    .prefetch(buffer_size=AUTOTUNE)
)

val_dataset = (
    val_dataset
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

# -----------------------------
# Data Augmentation
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
])
# -----------------------------
# MobileNetV2 Base Model
# -----------------------------
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False
# -----------------------------
# Build Model
# -----------------------------
inputs = tf.keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = Model(inputs, outputs)
# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
# -----------------------------
# Callbacks
# -----------------------------
callbacks = [

    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        "models/mango_disease_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1
    )

]
# -----------------------------
# Train Model
# -----------------------------
history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    callbacks=callbacks

)
# -----------------------------
# Save Final Model
# -----------------------------
model.save("models/mango_disease_model.keras")

print("\n✅ Training Completed Successfully!")
print("✅ Best model saved successfully.")
# -----------------------------
# Generate Predictions
# -----------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

y_true = []
y_pred = []
y_score = []

for images, labels in val_dataset:
    predictions = model.predict(images, verbose=0)

    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))
    y_score.extend(predictions)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_score = np.array(y_score)
# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    cmap="Blues",
    xticks_rotation=45,
    ax=ax,
    colorbar=False
)

plt.title("Confusion Matrix")
plt.tight_layout()

plt.savefig("results/confusion_matrix.png")
plt.close()

print("✅ Confusion Matrix saved.")


# -----------------------------
# ROC Curve
# -----------------------------
y_true_bin = label_binarize(
    y_true,
    classes=np.arange(num_classes)
)

plt.figure(figsize=(8, 6))

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(
        y_true_bin[:, i],
        y_score[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"{class_names[i]} (AUC = {roc_auc:.2f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend(loc="lower right", fontsize=8)

plt.tight_layout()

plt.savefig("results/roc_curve.png")
plt.close()

print("✅ ROC Curve saved.")

# -----------------------------
# Prediction Probability Histogram
# -----------------------------
import matplotlib.pyplot as plt

# Average prediction probability for each class
avg_probs = np.mean(y_score, axis=0)

plt.figure(figsize=(10, 5))

plt.bar(class_names, avg_probs)

plt.xticks(rotation=45)
plt.xlabel("Disease Classes")
plt.ylabel("Average Prediction Probability")
plt.title("Prediction Probability Histogram")

plt.tight_layout()

plt.savefig("results/prediction_histogram.png")
plt.close()

print("✅ Prediction Histogram saved.")