from model import Model
import dataset
import logging
from config import cfg

def train():
    logging.info("LOG - Load dataset")
    train_data, val_data = dataset.load_dataset(cfg.MODEL.IMG_DIR)

    logging.info("LOG - Training start...")
    model = Model(num_class=3 , augmentation=True)
    model.train_model(
        train_data=train_data,
        val_data=val_data, 
        epochs=cfg.MODEL.EPOCH,
        optimizer="adam")

    logging.info(f"LOG - Training finished !, saving model to {cfg.MODEL.SAVE_DIR}")
    model.save_model(save_dir=cfg.MODEL.SAVE_DIR)

    model.visualize(epochs=cfg.MODEL.EPOCH)

if __name__ == "__main__":
    train()