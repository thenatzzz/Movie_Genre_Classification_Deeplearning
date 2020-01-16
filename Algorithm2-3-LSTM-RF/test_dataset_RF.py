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
from keras.models import model_from_json
from keras.models import load_model
from joblib import dump, load
import tool
from sklearn.metrics import hamming_loss, confusion_matrix
from keras import backend as K
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve,roc_curve
from sklearn.metrics import average_precision_score

columns=['Action',	'Adventure',	'Animation',	'Comedy',	'Crime',	'Documentary',	'Drama',	'Family',	'Fantasy',	'Foreign',	'History',	'Horror',	'Music',	'Mystery',	'Romance',	'Science Fiction',	'TV Movie',	'Thriller',	'War',	'Western']
NUM_GENRES = 20
def cal_accuracy_at_least1(predict,y_test):
    accuracy = 0
    temp_array = predict * y_test
    row_with_all_zero = np.sum(~temp_array.any(1))
    return (len(y_test)-row_with_all_zero)/len(y_test)
def cal_accuracy_all(predict,y_test):
    return np.sum(np.all(predict == y_test, axis=1))/len(y_test)
def plot_recall_vs_precision(predict,y_test):
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    print(predict)
    print(y_test)
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
    plt.title("Precision vs. Recall curve (RF)")
    plt.savefig('rf_precision_vs_recall_curve.png')
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
    plt.title("ROC Curve (RF)")
    plt.savefig('rf_roc_curve.png')
    plt.show()
    plt.clf()

def main():
    df = pd.read_csv('test.csv', low_memory=False)

    # load json and create model
    model = load('randomForestModel.joblib')
    print("Loaded model from disk")

    # load test text
    df = df.drop(['imdb_id', 'title','overview','overview length','genres','labels_index'], axis = 1)
    xtest = df["features_content"].astype(str).apply(lambda x: tool.preprocessOverview(x))

    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=15000)
    xtest_tfidf = tfidf_vectorizer.fit_transform(xtest)


    preds = model.predict_proba(xtest_tfidf)
    original_predict = preds.copy() # make a copy

    preds[preds >= 0.4] = 1
    preds[preds < 0.4] = 0

    df_test = df.drop(['id','features_content'],axis=1)
    accuracy_at_least1 = cal_accuracy_at_least1(preds,df_test)
    accuracy_all = cal_accuracy_all(preds,df_test)

    print(list(df_test.columns))

    hammingloss = hamming_loss(preds,df_test)
    cm = confusion_matrix(preds.argmax(axis=1),df_test.values.argmax(axis=1))
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

    plot_recall_vs_precision(original_predict,df_test.values)
    plot_ROC_curve(original_predict,df_test.values)

    # preds_df = pd.DataFrame(data=preds[0:,0:])
    # print(preds_df.head())
    # preds_df.to_csv('test_result_RF.csv')
if __name__ == "__main__":
    main()
'''
RandomForestClassifier
['Action', 'Adventure', 'Animation', 'Comedy',          'Crime', 'Documentary',
'Drama',    'Family',    'Fantasy', 'Foreign',          'History', 'Horror',
'Music',    'Mystery',   'Romance', 'Science Fiction',  'TV Movie', 'Thriller',
'War', 'Western']

Precision:  [0.14164306 0.04012346 0.024      0.43686502 0.02484472 0.0297619
            0.31235539 0.         0.         0.         0.         0.
            0.         0.         0.01449275 0.         0.         0.00775194
            0.         0.        ]
Recall:  [0.14677104 0.0368272  0.03913043 0.25763195 0.04651163 0.09868421
          0.25862069 0.         0.         0.         0.         0.
          0.         0.         0.03225806 0.         0.         0.03448276
          0.         0.        ]
F1:  [0.14416141 0.03840468 0.02975202 0.32412056 0.03238862 0.04573167
      0.2829595  0.         0.         0.         0.         0.
      0.         0.         0.01999996 0.         0.         0.0126582
      0.         0.        ]
Accuracy at least one genre:  0.49406739439962033
Accuracy for all genre:  0.04235880398671096
Hamming_loss:  0.14563953488372092

Genre,    f1, recall,precision, number
Comedy,  0.32,  0.26,   0.44,    13131
Drama,   0.28, 0.26,   0.31,     20176
Action,  0.14, 0.15,  0.14,      6582
Documentary,0.05,0.10, 0.03,     3855
Horror,  0.04, 0.04, 0.04,       4668
'''
