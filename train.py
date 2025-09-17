from preprocess import get_datasets
from model_builder import build_simple_cnn
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

train_ds, val_ds = get_datasets('dataset/train', 'dataset/val', augment=True)
num_classes = len(train_ds.class_names)

model = build_simple_cnn(num_classes)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=4,
                    callbacks=[checkpoint, early_stop])

os.makedirs("model", exist_ok=True)
model.save("model")
print("Model saved to 'model/'")
