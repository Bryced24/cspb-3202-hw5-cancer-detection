import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


SEED = 42
IMG_SIZE = 64
BATCH_SIZE = 128
DATA_DIR = next(Path("/kaggle/input").rglob("train_labels.csv")).parent
OUTPUT_DIR = Path("/kaggle/working")
FIGURE_DIR = OUTPUT_DIR / "figures"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

labels = pd.read_csv(DATA_DIR / "train_labels.csv")
labels["path"] = labels["id"].map(lambda value: str(DATA_DIR / "train" / f"{value}.tif"))
test_paths = sorted((DATA_DIR / "test").glob("*.tif"))

print(f"Training rows: {len(labels):,}")
print(f"Test images: {len(test_paths):,}")
print(f"Duplicate training IDs: {labels['id'].duplicated().sum()}")
print(f"Missing training files: {(~labels['path'].map(os.path.exists)).sum()}")
print(labels["label"].value_counts().sort_index())
print(labels["label"].value_counts(normalize=True).sort_index().rename("proportion"))

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(6, 4))
counts = labels["label"].value_counts().sort_index()
ax.bar(["No tumor", "Tumor"], counts.values, color=["#4C78A8", "#E45756"])
ax.set_title("Training Label Counts")
ax.set_ylabel("Images")
for index, value in enumerate(counts.values):
    ax.text(index, value + 1500, f"{value:,}", ha="center")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class_distribution.png", dpi=160)
plt.show()

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for row, label_value in enumerate([0, 1]):
    sample_rows = labels[labels["label"] == label_value].sample(5, random_state=SEED)
    for col, path in enumerate(sample_rows["path"]):
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        axes[row, col].imshow(image)
        axes[row, col].axis("off")
        if col == 0:
            axes[row, col].set_title("No tumor" if label_value == 0 else "Tumor")
fig.suptitle("Example Training Patches")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "sample_images.png", dpi=160)
plt.show()

brightness_rows = []
for _, row in labels.sample(2000, random_state=SEED).iterrows():
    image = cv2.imread(row["path"])
    brightness_rows.append({"brightness": float(image.mean()), "label": row["label"]})
brightness = pd.DataFrame(brightness_rows)
fig, ax = plt.subplots(figsize=(7, 4))
for label_value, color, name in [(0, "#4C78A8", "No tumor"), (1, "#E45756", "Tumor")]:
    ax.hist(
        brightness.loc[brightness["label"] == label_value, "brightness"],
        bins=30,
        alpha=0.55,
        density=True,
        color=color,
        label=name,
    )
ax.set_title("Mean Pixel Brightness in a 2,000-Image Sample")
ax.set_xlabel("Mean pixel value")
ax.set_ylabel("Density")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "brightness_distribution.png", dpi=160)
plt.show()

model_rows, _ = train_test_split(
    labels,
    train_size=15000,
    random_state=SEED,
    stratify=labels["label"],
)
train_rows, validation_rows = train_test_split(
    model_rows,
    test_size=3000,
    random_state=SEED,
    stratify=model_rows["label"],
)
print(f"Training split: {len(train_rows):,}")
print(f"Validation split: {len(validation_rows):,}")


def read_image(path):
    file_path = path.numpy().decode("utf-8")
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return image.astype(np.float32) / 255.0


def load_labeled_image(path, label):
    image = tf.py_function(read_image, [path], tf.float32)
    image.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return image, tf.cast(label, tf.float32)


def load_unlabeled_image(path):
    image = tf.py_function(read_image, [path], tf.float32)
    image.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return image


def make_labeled_dataset(frame, training):
    dataset = tf.data.Dataset.from_tensor_slices((frame["path"].values, frame["label"].values))
    if training:
        dataset = dataset.shuffle(20000, seed=SEED, reshuffle_each_iteration=True)
    return (
        dataset.map(load_labeled_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


train_dataset = make_labeled_dataset(train_rows, training=True)
validation_dataset = make_labeled_dataset(validation_rows, training=False)


def build_model(use_augmentation=False, learning_rate=0.001, dropout=0.30):
    layers = [tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 3))]
    if use_augmentation:
        layers.extend(
            [
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.10),
            ]
        )
    layers.extend(
        [
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=1,
    restore_best_weights=True,
)

baseline_model = build_model()
baseline_history = baseline_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=2,
    callbacks=[early_stopping],
)

tuned_model = build_model(use_augmentation=True, learning_rate=0.0005, dropout=0.40)
tuned_history = tuned_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=3,
    callbacks=[early_stopping],
)


def validation_metrics(model):
    probabilities = model.predict(validation_dataset, verbose=1).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(validation_rows["label"], probabilities),
        "accuracy": accuracy_score(validation_rows["label"], predictions),
        "probabilities": probabilities,
        "predictions": predictions,
    }


baseline_metrics = validation_metrics(baseline_model)
tuned_metrics = validation_metrics(tuned_model)
results = pd.DataFrame(
    [
        {"model": "Baseline CNN", "validation_auc": baseline_metrics["auc"], "validation_accuracy": baseline_metrics["accuracy"]},
        {"model": "Augmented CNN", "validation_auc": tuned_metrics["auc"], "validation_accuracy": tuned_metrics["accuracy"]},
    ]
)
print(results)
results.to_csv(OUTPUT_DIR / "experiment_results.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for history, name in [(baseline_history, "Baseline"), (tuned_history, "Augmented")]:
    axes[0].plot(history.history["val_auc"], marker="o", label=name)
    axes[1].plot(history.history["val_loss"], marker="o", label=name)
axes[0].set_title("Validation ROC AUC")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("ROC AUC")
axes[1].set_title("Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Binary cross-entropy")
for ax in axes:
    ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "training_results.png", dpi=160)
plt.show()

matrix = confusion_matrix(validation_rows["label"], tuned_metrics["predictions"])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
ax.set_title("Augmented CNN Validation Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=160)
plt.show()

test_path_strings = np.array([str(path) for path in test_paths])
test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_path_strings)
    .map(load_unlabeled_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
best_model = baseline_model if baseline_metrics["auc"] >= tuned_metrics["auc"] else tuned_model
test_probabilities = best_model.predict(test_dataset, verbose=1).ravel()
submission = pd.DataFrame(
    {
        "id": [path.stem for path in test_paths],
        "label": test_probabilities,
    }
)
submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
best_model.save(OUTPUT_DIR / "best_cnn.keras")
print(submission.head())
print(f"Submission rows: {len(submission):,}")
