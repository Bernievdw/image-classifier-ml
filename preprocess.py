import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf

def split_dataset(dataset_dir, train_dir, val_dir, val_ratio=0.2):
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        images = os.listdir(class_path)
        train_imgs, val_imgs = train_test_split(images, test_size=val_ratio, random_state=42)

        for img in train_imgs:
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        
        for img in val_imgs:
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
            shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))

def get_datasets(train_dir, val_dir, img_size=(224,224), batch_size=32, augment=False):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch_size
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir, image_size=img_size, batch_size=batch_size
    )

    if augment:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1)
        ])
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds
