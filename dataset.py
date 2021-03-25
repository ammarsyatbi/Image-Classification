import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from config import cfg


AUTOTUNE = tf.data.AUTOTUNE

def load_dataset(data_dir):
    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(f"image count - {image_count}")
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(cfg.MODEL.IMG_HEIGHT, cfg.MODEL.IMG_WIDTH),
        batch_size=cfg.MODEL.BATCH_SIZE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(cfg.MODEL.IMG_HEIGHT, cfg.MODEL.IMG_WIDTH),
        batch_size=cfg.MODEL.BATCH_SIZE)


    # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Check shape
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    return train_ds, val_ds

# normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds))
# first_image = image_batch[0]

# # Notice the pixels values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))
    