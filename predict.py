import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from config import cfg
import utils
import json

CLASS_NAMES = utils.read_class_names(cfg.MODEL.CLASS_NAMES)
test_imgs = os.listdir(cfg.MODEL.TEST_DIR)
model = tf.keras.models.load_model(cfg.MODEL.SAVE_DIR)

predicted = {}
bad_imgs = []

for img_name in test_imgs:
    test_img_path = os.path.join(cfg.MODEL.TEST_DIR, img_name)
    try:
        img = keras.preprocessing.image.load_img(
        test_img_path, target_size=(cfg.MODEL.IMG_HEIGHT, cfg.MODEL.IMG_WIDTH)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        # load img
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(CLASS_NAMES[np.argmax(score)], 100 * np.max(score))
        )

        if CLASS_NAMES[np.argmax(score)] not in predicted:
            predicted[CLASS_NAMES[np.argmax(score)]] = []
        predicted[CLASS_NAMES[np.argmax(score)]].append(img_name)
    except Exception as e:
        print(e)
        bad_imgs.append(img_name)
        pass
    

print(f"Total bad images - {len(predicted)}")
with open(os.path.join(cfg.MODEL.RESULT_DIR, "predicted.json"), "w") as file:
    file.write(json.dumps(predicted))
