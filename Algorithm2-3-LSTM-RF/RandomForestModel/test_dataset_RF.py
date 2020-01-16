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
from sklearn.feature_extraction.text import TfidfVectorizer

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
preds[preds >= 0.4] = 1
preds[preds < 0.4] = 0


preds_df = pd.DataFrame(data=preds[0:,0:])

print(preds_df.head())

preds_df.to_csv('test_result_RF.csv')


