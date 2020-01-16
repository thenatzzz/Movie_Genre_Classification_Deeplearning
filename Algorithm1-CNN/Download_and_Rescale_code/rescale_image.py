import keras
from keras.preprocessing import image
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from threading import Thread
import gc
import os
import importlib
from keras import backend as K

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
''' Code to re-load original poster images from dir and rescale to specific pixel.
  Then save to new dir (since original posters are too big) '''

def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend
    if backend == "tensorflow":
        K.get_session().close()
        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        K.set_session(K.tf.Session(config=cfg))
        K.clear_session()

set_keras_backend("tensorflow")
PIXEL = 224 #ResNet50 size

SIZE_1 = 8550
SIZE_2 = 8550
SIZE_3 = 8550
SIZE_4 = 8550
SIZE_5 = 8582
SIZE = SIZE_1+SIZE_2+SIZE_3+SIZE_4+SIZE_5
train = pd.read_csv('new_clean_data_with_path.csv')    # reading the csv file

LOAD_DIR = 'poster_images/'
SAVE_DIR = 'transformed_poster_images2/'

def main1():
    for i in tqdm(range(0,SIZE_1)):
        img = image.load_img(LOAD_DIR+train['imdb_id'][i]+'.jpg',target_size=(PIXEL,PIXEL,3))
        img = image.img_to_array(img)
        img = img/255
        image.save_img(SAVE_DIR+train['imdb_id'][i]+'.jpg',img)
def main2():
    for i in tqdm(range(SIZE_1,SIZE_1+SIZE_2)):
        img = image.load_img(LOAD_DIR+train['imdb_id'][i]+'.jpg',target_size=(PIXEL,PIXEL,3))
        img = image.img_to_array(img)
        img = img/255
        image.save_img(SAVE_DIR+train['imdb_id'][i]+'.jpg',img)
def main3():
    for i in tqdm(range(SIZE_1+SIZE_2,SIZE_1+SIZE_2+SIZE_3)):
        img = image.load_img(LOAD_DIR+train['imdb_id'][i]+'.jpg',target_size=(PIXEL,PIXEL,3))
        img = image.img_to_array(img)
        img = img/255
        image.save_img(SAVE_DIR+train['imdb_id'][i]+'.jpg',img)
def main4():
    for i in tqdm(range(SIZE_1+SIZE_2+SIZE_3,SIZE_1+SIZE_2+SIZE_3+SIZE_4)):
        img = image.load_img(LOAD_DIR+train['imdb_id'][i]+'.jpg',target_size=(PIXEL,PIXEL,3))
        img = image.img_to_array(img)
        img = img/255
        image.save_img(SAVE_DIR+train['imdb_id'][i]+'.jpg',img)
def main5():
    for i in tqdm(range(SIZE_1+SIZE_2+SIZE_3+SIZE_4,SIZE_1+SIZE_2+SIZE_3+SIZE_4+SIZE_5)):
        img = image.load_img(LOAD_DIR+train['imdb_id'][i]+'.jpg',target_size=(PIXEL,PIXEL,3))
        img = image.img_to_array(img)
        img = img/255
        image.save_img(SAVE_DIR+train['imdb_id'][i]+'.jpg',img)

def main():
    gc.collect()

if __name__ == '__main__':
    t1 = Thread(target = main1)
    t2 = Thread(target = main2)
    t3 = Thread(target = main3)
    t4 = Thread(target = main4)
    t5 = Thread(target = main5)

    t = Thread(target = main)

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    t.start()
