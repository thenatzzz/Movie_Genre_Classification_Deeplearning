from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras import regularizers, optimizers
from keras.layers import Input
from keras.preprocessing import image
from sklearn.metrics import hamming_loss, confusion_matrix
import matplotlib.pyplot as plt
import keras
from keras.models import Model, load_model
from keras.applications import MobileNet,ResNet50
from PIL import ImageFile
from sklearn.metrics import precision_recall_curve,roc_curve
from sklearn.metrics import average_precision_score
from keras import backend as K

ImageFile.LOAD_TRUNCATED_IMAGES = True

import tensorflow as tf
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

import pandas as pd
import numpy as np

df=pd.read_csv('new_clean_data_with_path.csv')
columns=["Music","Western","Thriller",	"Adventure","Drama"	,"Mystery"	,"TV Movie"	,"Crime"	,"Fantasy"	,"Action"	,"Animation"	,"Romance"	,"History",	"Horror",	"War",	"Family",	"Documentary"	,"Comedy"	,"Foreign",	"Science Fiction"]

# columns=["Thriller",	"Adventure","Drama"	,"Mystery"	,"Crime"	,"Fantasy"	,"Action"	,"Animation"	,"Romance"	,	"Horror",		"Family",	"Documentary"	,"Comedy"	,"Foreign",	"Science Fiction"]
# df = df.drop(["TV Movie","Music","Western","War","History"],axis=1)

df['imdb_id']= df['imdb_id'].astype(str)+'.jpg'
df = df.sample(frac=1)
#1,2,13,15
NUM_GENRES = 20
PIXEL=100 #224

SIZE_TRAIN=30000
SIZE_VALIDATION = 8000
SIZE_TEST = 4000

datagen=ImageDataGenerator(rescale=1./255.)
test_datagen=ImageDataGenerator(rescale=1./255.)
train_generator=datagen.flow_from_dataframe(
                                                dataframe=df[:SIZE_TRAIN],
                                                # directory="poster/poster_images",
                                                directory="poster/transformed_poster_images",
                                                x_col="imdb_id",
                                                y_col=columns,
                                                batch_size=128,
                                                seed=42,
                                                shuffle=True,
                                                class_mode="other",
                                                target_size=(PIXEL,PIXEL))

valid_generator=test_datagen.flow_from_dataframe(
                                                    dataframe=df[SIZE_TRAIN:SIZE_TRAIN+SIZE_VALIDATION],
                                                    # directory="poster/poster_images",
                                                    directory="poster/transformed_poster_images",
                                                    x_col="imdb_id",
                                                    y_col=columns,
                                                    batch_size=128,
                                                    seed=42,
                                                    shuffle=True,
                                                    class_mode="other",
                                                    target_size=(PIXEL,PIXEL))
test_generator=test_datagen.flow_from_dataframe(
                                                    dataframe=df[SIZE_TRAIN+SIZE_VALIDATION:SIZE_TRAIN+SIZE_VALIDATION+SIZE_TEST],
                                                    # directory="poster/poster_images",
                                                    directory="poster/transformed_poster_images",
                                                    x_col="imdb_id",
                                                    batch_size=50,
                                                    seed=42,
                                                    shuffle=False,
                                                    class_mode=None,
                                                    target_size=(PIXEL,PIXEL))
def cal_accuracy_at_least1(predict,y_test):
    accuracy = 0
    temp_array = predict * y_test
    row_with_all_zero = np.sum(~temp_array.any(1))
    return (len(y_test)-row_with_all_zero)/len(y_test)
def cal_accuracy_all(predict,y_test):
    return np.sum(np.all(predict == y_test, axis=1))/len(y_test)
def create_model_1():
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
def create_model_2():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(PIXEL,PIXEL,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_GENRES, activation='sigmoid'))
    return model
def create_MobileNet(models='MobileNet'):
    if models == 'MobileNet' and PIXEL==100:
        model_weight = None
        base_model=MobileNet(weights=model_weight,include_top=False,input_shape=(PIXEL,PIXEL,3)) #imports the mobilenet model and discards the last 1000 neuron layer.
    elif PIXEL==224:
        model_weight = 'imagenet'
        base_model=ResNet50(weights=model_weight,include_top=False,input_shape=(PIXEL,PIXEL,3)) #imports the resnet50 model and discards the last neuron layer.
    # base_model=MobileNet(weights=model_weight,include_top=False,input_shape=(PIXEL,PIXEL,3)) #imports the mobilenet model and discards the last 1000 neuron layer.
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3
    preds=Dense(NUM_GENRES,activation='sigmoid')(x)
    return Model(inputs=base_model.input,outputs=preds)
def plot_recall_vs_precision(predict,y_test):
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(NUM_GENRES):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            predict[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], predict[:, i])
        plt.plot(recall[i], precision[i], lw=2, label=columns[i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),predict.ravel())
    average_precision["micro"] = average_precision_score(y_test, predict,average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(b=True, which='major', color='g', linestyle='--')

    # Put a legend to the right of the current axis
    l1 = plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    plt.subplots_adjust(right=0.7)
    plt.title("Precision vs. Recall curve (CNN)")
    plt.savefig('cnn_precision_vs_recall_curve.png')
    plt.show()
    plt.clf()
def plot_ROC_curve(predict,y_test):
    # roc curve
    fpr = dict()
    tpr = dict()
    for i in range(NUM_GENRES):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i],  predict[:, i])
        plt.plot(fpr[i], tpr[i], lw=2, label=columns[i])

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.grid(b=True, which='major', color='g', linestyle='--')
    # plt.legend(loc="best")
    # Put a legend to the right of the current axis
    l1 = plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    plt.subplots_adjust(right=0.7)
    plt.title("ROC Curve (CNN)")
    plt.savefig('cnn_roc_curve.png')
    plt.show()
    plt.clf()

def main():
    ''' Specify models to be used '''
    ''' ####################### TRAIN CODE ##################################'''
    #model = create_model_1()
    model = create_model_2()
    # model = create_MobileNet()

    ''' Specify optimizers and other hyperparameters'''
    # model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="binary_crossentropy",metrics=["accuracy"])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    '''
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=10,
                        # epochs=2,
                        workers=6,
                        use_multiprocessing=False
    )
    model.save("cnn_model_2.h5")
    '''
    ####################### TEST CODE #####################################'''
    model = load_model('cnn_model_2.h5')
    predict = model.predict_generator(test_generator,steps = STEP_SIZE_TEST)
    original_predict = predict.copy() # make a copy

    predict[predict>=0.4] = 1.0
    predict[predict<0.4] = 0.0

    y_test = np.array(df[SIZE_TRAIN+SIZE_VALIDATION:SIZE_TRAIN+SIZE_VALIDATION+SIZE_TEST].drop(['imdb_id', 'genres','poster_path'],axis=1))


    accuracy_at_least1 = cal_accuracy_at_least1(predict,y_test)
    accuracy_all = cal_accuracy_all(predict,y_test)
    hammingloss = hamming_loss(predict,y_test)

    cm = confusion_matrix(predict.argmax(axis=1),y_test.argmax(axis=1))
    recall_ = np.diag(cm) / (np.sum(cm, axis = 1) + K.epsilon())
    precision_ = np.diag(cm) / (np.sum(cm, axis = 0)+ K.epsilon())
    f1_ =  (2*recall_*precision_)/(precision_+recall_+ K.epsilon())

    # print("Confusion matrx: ",cm)
    print("Precision: ",precision_)
    print("Recall: ",recall_)
    print("F1: ",f1_)
    print("Accuracy at least one genre: ",accuracy_at_least1)
    print("Accuracy for all genre: ",accuracy_all)
    print("Hamming_loss: ",hammingloss)

    plot_recall_vs_precision(original_predict,y_test)
    plot_ROC_curve(original_predict,y_test)

if __name__ == '__main__':
    main()



'''
CNN
columns=["Music",  "Western","Thriller", "Adventure","Drama"	,"Mystery"
       ,"TV Movie","Crime"	,"Fantasy"	,"Action"	,"Animation","Romance"
       ,"History", "Horror", "War",     "Family",	"Documentary","Comedy"
       ,"Foreign",	"Science Fiction"]
Numbers:  1590,    1038,      7600,      3486,       20176,     2457,
         758,      4292,      2304,      6582,       1919,      6711,
        1394,      4668,      1322,      2763,       3855,     13131,
        1614,      3038
Precision:  [0.0979021  0.02666667 0.24753868 0.04651163 0.70468859 0.02272727
             0.         0.015625   0.         0.04065041 0.16923077 0.07751938
             0.         0.07361963 0.         0.06521739 0.0502092  0.21818182
             0.         0.        ]
Recall:  [0.04268293 0.66666664 0.39461883 0.21818182 0.42453626 0.49999998
          0.         0.11111111 0.         0.07352941 0.24444444 0.14492754
          0.         0.26666667 0.         0.07317073 0.25       0.18625277
          0.         0.        ]
F1:  [0.05944794 0.05128204 0.30423504 0.07667729 0.52986052 0.04347825
      0.         0.02739724 0.         0.05235598 0.19999995 0.10101006
      0.         0.11538458 0.         0.06896547 0.08362367 0.20095689
      0.         0.        ]
Accuracy at least one genre:  0.58375
Accuracy for all genre:  0.10375
Hamming_loss:  0.1102

loss: 0.2132 - acc: 0.9148 -
val_ loss: 0.3031 - val_acc: 0.8946

Genre,    f1, recall,precision, number
Drama,  0.53,  0.42,   0.70,    20176
Thriller,0.30, 0.39,   0.25,    7600
Comedy,  0.21, 0.19,  0.22,    13131
Animation,0.20,0.24, 0.17,     1919
Horror,  0.12, 0.27, 0.07,    4668




Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 100, 100, 32)      896
_________________________________________________________________
activation_1 (Activation)    (None, 100, 100, 32)      0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 98, 98, 32)        9248
_________________________________________________________________
activation_2 (Activation)    (None, 98, 98, 32)        0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 49, 49, 32)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 49, 49, 32)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 49, 49, 64)        18496
_________________________________________________________________
activation_3 (Activation)    (None, 49, 49, 64)        0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 47, 47, 64)        36928
_________________________________________________________________
activation_4 (Activation)    (None, 47, 47, 64)        0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 23, 23, 64)        0
_________________________________________________________________
dropout_2 (Dropout)          (None, 23, 23, 64)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 33856)             0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               17334784
_________________________________________________________________
activation_5 (Activation)    (None, 512)               0
_________________________________________________________________
dropout_3 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 20)                10260
=================================================================
Total params: 17,410,612
Trainable params: 17,410,612






'''
