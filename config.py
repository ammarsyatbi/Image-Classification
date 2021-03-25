#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict

__C = edict()
cfg = __C
# Model option
__C.MODEL = edict()

# Config
__C.MODEL.IMG_HEIGHT = 300
__C.MODEL.IMG_WIDTH = 300
__C.MODEL.BATCH_SIZE =  16
__C.MODEL.EPOCH = 1


__C.MODEL.IMG_DIR = "./data/image"
__C.MODEL.SAVE_DIR = "./data/model"
__C.MODEL.CKPT_DIR = "./data/model/checkpoint"
__C.MODEL.RESULT_DIR = "./data/result"
__C.MODEL.TEST_DIR = "./data/test"
__C.MODEL.CLASS_NAMES = "./data/classes/classes.names"