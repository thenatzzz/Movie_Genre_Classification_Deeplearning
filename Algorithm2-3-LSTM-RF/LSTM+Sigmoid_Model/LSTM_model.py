import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import os

def preprocessOverview(text):
    # tokenize text and set to lower case
    tokens = [x.strip().lower() for x in nltk.word_tokenize(text)]

    # get stopwords from nltk, remove them from our list as well as all punctuations and numbers
    stop_words = stopwords.words('english')
    output = [word for word in tokens if (word not in stop_words and word.isalpha())]

    return " ".join(output)


def sentences_to_indices(X, word_to_index, max_len):

    # number of training examples
    m = X.shape[0]

    # Initialize X_indices as a numpy matrix of zeros and the correct shape (~ 1 line)
    X_indices = np.zeros((m, max_len))

    # loop over training examples
    for i in range(m):

        # Convert the ith training sentence in lower case and split is into words -> get a list of words.
        sentence_words = [x.lower() for x in X[i].split()]

        # Initialize j to 0
        j = 0

        # Loop over the words in sentence_words
        for w in sentence_words:

            # check that the word is within our GloVe dataset, otherwise pass
            if w in word_to_index.keys():
                # Set the (i,j)th entry of X_indices to the index of the correct word.
                X_indices[i, j] = word_to_index[w]

                # Increment j to j + 1
                j = j + 1
            else:
                pass

    return X_indices

def read_glove_vecs_only_alpha(glove_file):
    with open(glove_file, 'r', encoding='utf8') as f:

        words = set()
        word_to_vec_map = {}

        for line in f:
            line = line.strip().split()
            curr_word = line[0]

            # only consider words containing alphabetical letters
            if curr_word.isalpha():
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}

        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1

    return words_to_index, index_to_words, word_to_vec_map




df = pd.read_csv('train.csv', low_memory=False)
max_sequence_length = df["overview length"].max()
df = df.drop(['imdb_id', 'title','overview','overview length','genres','labels_index'], axis = 1)
df["features_content"] = df["features_content"].astype(str).apply(lambda x: preprocessOverview(x))

Y = df[df.columns[2:]]
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs_only_alpha('glove.6B.100d.txt')

X = df['features_content']
X_indices = sentences_to_indices(X, word_to_index, max_sequence_length)


xtrain, xval, ytrain, yval = train_test_split(X_indices, Y, test_size=0.2, random_state=9)

top_words = 500

x_train = sequence.pad_sequences(xtrain, maxlen=max_sequence_length)
x_val = sequence.pad_sequences(xval, maxlen=max_sequence_length)


embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_sequence_length))
model.add(LSTM(100))
model.add(Dense(20, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, ytrain, epochs=20, batch_size=64)


scores = model.evaluate(x_val, yval, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



model_json = model.to_json()
with open(r"C:\Users\dasan\Desktop\P726(2)\model_LSTM.json", "w+") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(r"C:\Users\dasan\Desktop\P726(2)\model_LSTM.h5")
