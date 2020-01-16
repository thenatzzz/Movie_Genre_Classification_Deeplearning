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
import tool

test = pd.read_csv('test.csv', low_memory=False)

# load json and create model
json_file = open('model_LSTM.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model_LSTM.h5")
print("Loaded model from disk")

loaded_model.save('model_LSTM.h5')
model=load_model('model_LSTM.h5')

# load test text
max_sequence_length = 148
test = test.drop(['imdb_id', 'title','overview','overview length','genres','labels_index'], axis = 1)
test["features_content"] = test["features_content"].astype(str).apply(lambda x: tool.preprocessOverview(x))
word_to_index, index_to_word, word_to_vec_map = tool.read_glove_vecs_only_alpha('glove.6B.100d.txt')
X = test['features_content']
x_indices = tool.sentences_to_indices(X, word_to_index, max_sequence_length)
x_indices = tool.sequence.pad_sequences(x_indices, maxlen=max_sequence_length)

preds = model.predict(x_indices)
preds[preds >= 0.2] = 1
preds[preds < 0.2] = 0


preds_df = pd.DataFrame(data=preds[0:,0:])

print(preds_df.head())

preds_df.to_csv('test_result_LSTM.csv')


