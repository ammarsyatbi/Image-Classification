import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from config import cfg
import json
import os

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(cfg.MODEL.IMG_HEIGHT, 
                                                              cfg.MODEL.IMG_WIDTH,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

class Model():
    def __init__(self, num_class, augmentation):
        self.num_classes = num_class
        self.augmentation = True

        if augmentation:
            self.model = Sequential([
                            data_augmentation,
                            layers.experimental.preprocessing.Rescaling(1./255),
                            layers.Conv2D(16, 3, padding='same', activation='relu'),
                            layers.MaxPooling2D(),
                            layers.Conv2D(32, 3, padding='same', activation='relu'),
                            layers.MaxPooling2D(),
                            layers.Conv2D(64, 3, padding='same', activation='relu'),
                            layers.MaxPooling2D(),
                            layers.Dropout(0.2),
                            layers.Flatten(),
                            layers.Dense(128, activation='relu'),
                            layers.Dense(self.num_classes)
                        ])
        else:
            self.model = Sequential([
                        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(cfg.MODEL.IMG_HEIGHT, cfg.MODEL.IMG_WIDTH, 3)),
                        layers.Conv2D(16, 3, padding='same', activation='relu'),
                        layers.MaxPooling2D(),
                        layers.Conv2D(32, 3, padding='same', activation='relu'),
                        layers.MaxPooling2D(),
                        layers.Conv2D(64, 3, padding='same', activation='relu'),
                        layers.MaxPooling2D(),
                        layers.Flatten(),
                        layers.Dense(128, activation='relu'),
                        layers.Dense(self.num_classes)
                    ])
        print(self.model.summary())
        

    def train_model(self, train_data, val_data, epochs=10, optimizer='adam', checkpoint_filepath="./data/model/checkpoint/ckpt"):
        mdl_ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
                        filepath=checkpoint_filepath,
                        monitor='val_accuracy',
                        mode='max',
                        save_best_only=True)

        self.model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        self.history = self.model.fit(
                                train_data,
                                validation_data=val_data,
                                epochs=epochs,
                                callbacks = [mdl_ckpt_cb]
                            )
        with open(os.path.join(cfg.MODEL.SAVE_DIR, "history.json"), "w") as hist_file:
            hist_file.write(json.dumps(cfg.MODEL.SAVE_DIR))
        
    def save_model(self, save_dir=cfg.MODEL.SAVE_DIR):
        self.model.save(save_dir)

    def load_model(self, save_dir=cfg.MODEL.SAVE_DIR):
         self.model = tf.keras.models.load_model(cfg.MODEL.SAVE_DIR)

    def visualize(self, save_path=os.path.join(cfg.MODEL.RESULT_DIR,"eval.png"), epochs=10):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig(save_path)