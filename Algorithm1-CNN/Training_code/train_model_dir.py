import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
from PIL import ImageFile
from random import randrange

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 2} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

ImageFile.LOAD_TRUNCATED_IMAGES = True

SIZE = 4000
PIXEL = 100
NUM_GENRES = 20

def create_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(PIXEL,PIXEL,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_GENRES, activation='sigmoid'))
    return model
#1000,1.39
def main():
    train = pd.read_csv('new_clean_data_with_path.csv')    # reading the csv file
    train = train.sample(frac=1)
    print(train.head())
    print(train.columns)

    train_image = []
    # for i in tqdm(range(10)):
    for i in range(SIZE):
        # img = image.load_img('poster_images/'+train['imdb_id'][i]+'.jpg',target_size=(PIXEL,PIXEL,3))
        img = image.load_img('transformed_poster_images/'+train['imdb_id'][i]+'.jpg',target_size=(PIXEL,PIXEL,3))
        img = image.img_to_array(img)
        img = img/255
        # image.save_img('transformed_poster_images/'+train['imdb_id'][i]+'.jpg',img)
        train_image.append(img)
        print("Load img:",i, " of total ",SIZE)
    X = np.array(train_image)
    y = np.array(train[:SIZE].drop(['imdb_id', 'genres','poster_path'],axis=1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

    model = create_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64,verbose=2)

    poster_location = 'poster_images/'
    sample_img = poster_location+train['imdb_id'][40001]+'.jpg'
    img = image.load_img(sample_img,target_size=(PIXEL,PIXEL,3))
    img = image.img_to_array(img)
    img = img/255
    classes = np.array(train.columns[3:])
    proba = model.predict(img.reshape(1,PIXEL,PIXEL,3))
    top_3 = np.argsort(proba[0])[:-4:-1]
    for i in range(3):
        print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))
    plt.imshow(img)
    # plt.show()

if __name__ == '__main__':
    main()
