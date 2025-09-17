import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=4, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--output_dir', type=str, default='model', help='Directory to save model')
parser.add_argument('--dataset_path', type=str, default=r"C:\Users\Bernard\Downloads\PetImages",
                    help='Path to local PetImages folder')
args = parser.parse_args()

train_ds = image_dataset_from_directory(
    args.dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),  
    batch_size=args.batch_size
)

val_ds = image_dataset_from_directory(
    args.dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=args.batch_size
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')  
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs
)

os.makedirs(args.output_dir, exist_ok=True)
model.save(args.output_dir)

print(f"Model saved to {args.output_dir}")
