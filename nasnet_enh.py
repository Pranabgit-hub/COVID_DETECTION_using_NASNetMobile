# ==============================================================
# NASNetMobile with Histogram Equalization + Image Complement
# ==============================================================
import os
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ==============================================================
# 1️⃣ Apply Histogram Equalization + Image Complement
# ==============================================================

def preprocess_and_save_images(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for cls in ["Covid", "Normal"]:
        src_class_dir = os.path.join(src_dir, cls)
        dest_class_dir = os.path.join(dest_dir, cls)
        os.makedirs(dest_class_dir, exist_ok=True)
        
        for img_name in os.listdir(src_class_dir):
            if img_name.startswith('.'):
                continue
            img_path = os.path.join(src_class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (224, 224))
            
            # --- Histogram Equalization per channel ---
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)
            
            # --- Image Complement ---
            img_complement = 255 - img_eq
            
            # Save processed image
            save_path = os.path.join(dest_class_dir, img_name)
            cv2.imwrite(save_path, img_complement)

# Source dataset
base_dir = "Dataset_A"
preprocessed_dir = "Dataset_A_Preprocessed"

print("⚙️ Applying Histogram Equalization + Image Complement ...")
preprocess_and_save_images(base_dir, preprocessed_dir)
print("✅ Preprocessing complete!")

# ==============================================================
# 2️⃣ Split Dataset (70/15/15)
# ==============================================================

split_dir = "/content/Dataset_Split_Processed"
classes = ["Covid", "Normal"]

for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(split_dir, split, cls), exist_ok=True)

for cls in classes:
    cls_path = os.path.join(preprocessed_dir, cls)
    images = [f for f in os.listdir(cls_path) if not f.startswith('.')]
    
    train_files, temp_files = train_test_split(images, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    
    for f in train_files:
        shutil.copy(os.path.join(cls_path, f), os.path.join(split_dir, "train", cls))
    for f in val_files:
        shutil.copy(os.path.join(cls_path, f), os.path.join(split_dir, "val", cls))
    for f in test_files:
        shutil.copy(os.path.join(cls_path, f), os.path.join(split_dir, "test", cls))

print("✅ Dataset split complete!")

# ==============================================================
# 3️⃣ Data Generators
# ==============================================================

img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    os.path.join(split_dir, "train"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_gen = val_test_datagen.flow_from_directory(
    os.path.join(split_dir, "val"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_gen = val_test_datagen.flow_from_directory(
    os.path.join(split_dir, "test"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# ==============================================================
# 4️⃣ Build NASNetMobile Model
# ==============================================================

base_model = NASNetMobile(
    weights='imagenet',
    include_top=False,
    input_shape=(img_size[0], img_size[1], 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==============================================================
# 5️⃣ Training Setup
# ==============================================================

checkpoint_path = "/content/nasnet_preprocessed_best.h5"

checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ==============================================================
# 6️⃣ Train Model (10 epochs)
# ==============================================================

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[checkpoint, early_stop]
)

# ==============================================================
# 7️⃣ Evaluate on Test Set
# ==============================================================

model.load_weights(checkpoint_path)

test_loss, test_acc = model.evaluate(test_gen)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")
print(f"✅ Test Loss: {test_loss:.4f}")

preds = model.predict(test_gen)
y_pred = (preds > 0.5).astype(int)
y_true = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ==============================================================
# 8️⃣ Plot Accuracy & Loss
# ==============================================================

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy (Train vs Val)')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss (Train vs Val)')
plt.legend()
plt.tight_layout()
plt.show()
