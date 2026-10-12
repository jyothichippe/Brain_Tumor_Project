
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# -------------------------------
# 1. Configuration
# -------------------------------

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

# -------------------------------
# 2. Load training and validation data
# -------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)

if num_classes < 2:
    raise ValueError("The dataset must contain at least two classes.")

# Save the class order for the Streamlit application
with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)

# -------------------------------
# 3. Load test data
# -------------------------------

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

if test_ds.class_names != class_names:
    raise ValueError(
        "Training and testing class names or order do not match."
    )

# -------------------------------
# 4. Data augmentation
# -------------------------------

augmentation = models.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.05, 0.05)
], name="data_augmentation")

# Efficient input pipeline
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(
    1000, seed=SEED
).prefetch(AUTOTUNE)

val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# -------------------------------
# 5. Create custom CNN
# -------------------------------

def build_cnn():
    inputs = layers.Input(shape=(224, 224, 3))

    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    x = layers.Conv2D(
        32, 3, padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        64, 3, padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        128, 3, padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(
        num_classes, activation="softmax"
    )(x)

    return models.Model(inputs, outputs, name="Custom_CNN")


# -------------------------------
# 6. Create transfer-learning model
# -------------------------------

def build_transfer_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze pretrained layers
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))

    x = augmentation(inputs)

    # Convert pixel values from [0, 255] to [-1, 1]
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0
    )(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    return models.Model(
        inputs,
        outputs,
        name="MobileNetV2_Transfer"
    )




# -------------------------------
# 7. Train and evaluate each model
# -------------------------------



def run_experiment(model, model_name):
    print("\nTraining:", model_name)

    model_path = f"models/{model_name}.keras"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Load the best saved model
    best_model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    # Evaluate on test data
    y_true = np.concatenate([
        labels.numpy() for _, labels in test_ds
    ])

    probabilities = best_model.predict(
        test_ds,
        verbose=0
    )
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    print("\nClassification Report:", model_name)

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        target_names=class_names,
        zero_division=0
    )

    print(report)

    with open(
        f"results/{model_name}_classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    # 1. Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes)
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.tight_layout()

    plt.savefig(
        f"results/{model_name}_confusion_matrix.png",
        dpi=150
    )
    plt.close()

    # 2. Accuracy graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title(f"{model_name} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"results/{model_name}_accuracy.png",
        dpi=150
    )
    plt.close()

    # 3. Loss graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title(f"{model_name} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"results/{model_name}_loss.png",
        dpi=150
    )
    plt.close()

    print(f"Saved graphs for {model_name}")

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision_Macro": precision,
        "Recall_Macro": recall,
        "F1_Macro": f1
    }



# -------------------------------
# 8. Run both experiments
# -------------------------------


# Train both models
cnn = build_cnn()
cnn_result = run_experiment(cnn, "custom_cnn")

transfer_model = build_transfer_model()
transfer_result = run_experiment(
    transfer_model,
    "mobilenetv2"
)

# Compare evaluation metrics
comparison = pd.DataFrame([
    cnn_result,
    transfer_result
])

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

best_row = comparison.loc[
    comparison["F1_Macro"].idxmax()
]

print(
    "\nBest model by macro F1-score:",
    best_row["Model"]
)

with open(
    "results/model_comparison.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(comparison.to_string(index=False))
    file.write(
        "\n\nBest model by macro F1-score: "
        + str(best_row["Model"])
    )

print("\nTraining and evaluation completed.")
print("Check the models/ and results/ folders.")


# -------------------------------
# 9. Compare the models
# -------------------------------

comparison = pd.DataFrame([
    cnn_result,
    transfer_result
])

comparison.to_csv(
    "results/model_comparison.csv", index=False
)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

best_row = comparison.loc[
    comparison["F1_Macro"].idxmax()
]

print("\nBest model by macro F1-score:", best_row["Model"])

# Save a readable text report
with open("results/model_comparison.txt", "w") as f:
    f.write(comparison.to_string(index=False))
    f.write(
        "\n\nBest model by macro F1-score: "
        + str(best_row["Model"])
    )

print("\nTraining and evaluation completed.")
print("Check the models/ and results/ folders.")

# Create results folder if it does not exist
os.makedirs("results", exist_ok=True)

