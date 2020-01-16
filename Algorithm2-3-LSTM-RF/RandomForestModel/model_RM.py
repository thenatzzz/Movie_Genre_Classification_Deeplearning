import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from joblib import dump, load

def preprocessOverview(text):
    """
    Function that preprocesses sentence by setting to lower case, removing stop words and removing
    words that contain characters that are not letters

    Arguments:
    text -- overview of the movie

    Returns:
    output -- preprocessed overview
    """

    # tokenize text and set to lower case
    tokens = [x.strip().lower() for x in nltk.word_tokenize(text)]

    # get stopwords from nltk, remove them from our list as well as all punctuations and numbers
    stop_words = stopwords.words('english')
    output = [word for word in tokens if (word not in stop_words and word.isalpha())]

    return " ".join(output)


df = pd.read_csv('train.csv', low_memory=False)
df = df.drop(['imdb_id', 'title','overview','overview length','genres','labels_index'], axis = 1)
df["features_content"] = df["features_content"].astype(str).apply(lambda x: preprocessOverview(x))

Y = df[df.columns[2:]]


xtrain, xval, ytrain, yval = train_test_split(df['features_content'], Y, test_size=0.2, random_state=9)


tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=15000)
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)




forest = RandomForestClassifier(n_estimators=10, random_state=1)
model = OneVsRestClassifier(forest, n_jobs=-1)
model.fit(xtrain_tfidf, ytrain)



dump(model, 'randomForestModel.joblib')
