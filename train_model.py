# Final, validated code for coffee bean classifier model training
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks

# Paths
train_dir = "train"
test_dir = "test"
img_size = (128, 128)
batch_size = 32
num_classes = 4  # Dark, Green, Light, Medium

# Data generators with augmentation for training, only rescale for test
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.1,  # Use 10% of training data for validation
)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Use subfolders as class labels
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42,
)
val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=True,
    seed=42,
)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)

# Print class indices for reference
print("Class indices:", train_gen.class_indices)

# Model definition
model = models.Sequential(
    [
        layers.Input(shape=(*img_size, 3)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks for best model saving and early stopping
checkpoint = callbacks.ModelCheckpoint(
    "coffee_bean_classifier.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)
earlystop = callbacks.EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True
)

# Training
epochs = 25
history = model.fit(
    train_gen, epochs=epochs, validation_data=val_gen, callbacks=[checkpoint, earlystop]
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_gen)
print(f"Test accuracy: {test_acc*100:.2f}%")

# Save final model (best weights already saved by checkpoint)
print("Model training complete. Best model saved as coffee_bean_classifier.h5")
