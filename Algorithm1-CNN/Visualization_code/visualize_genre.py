import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_all_genre(df,cat_columns):
    sns.set(font_scale = 2)
    plt.figure(figsize=(15,8))
    ax= sns.barplot(cat_columns, df.iloc[:,3:].sum().values)
    plt.title("Movie in each genre", fontsize=24)
    plt.ylabel('Number of movies', fontsize=18)
    plt.xlabel('Movie Genres ', fontsize=18)
    rects = ax.patches
    labels = df.iloc[:,3:].sum().values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)
    plt.xticks(rotation=90)
    plt.grid(b=True, which='major', color='g', linestyle='--')
    plt.tight_layout()
    plt.show()
def visualize_movies_genres(df,cat_columns):
    sum_row_category = df.iloc[:,3:].sum(axis=1)
    multiLabel_counts = sum_row_category.value_counts()
    sns.set(font_scale = 2)
    plt.figure(figsize=(15,8))
    ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)
    plt.title("Movies having multiple genres ")
    plt.ylabel('Number of movies', fontsize=18)
    plt.xlabel('Number of genres', fontsize=18)
    rects = ax.patches
    labels = multiLabel_counts.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    plt.grid(b=True, which='major', color='g', linestyle='--')
    plt.show()

def main():
    df=pd.read_csv('new_clean_data_with_path.csv')
    cat_columns=["Music","Western","Thriller",	"Adventure","Drama"	,"Mystery"	,"TV Movie"	,"Crime"	,"Fantasy"	,"Action"	,"Animation"	,"Romance"	,"History",	"Horror",	"War",	"Family",	"Documentary"	,"Comedy"	,"Foreign",	"Science Fiction"]
    category_col = np.array(df.drop(['imdb_id', 'genres','poster_path'],axis=1))
    print("Number of rows in data =",df.shape[0])
    print("Number of columns in data =",df.shape[1])
    print("Number of categories in data =",len(cat_columns))
    print("\n")
    print("**Sample data:**")

    visualize_all_genre(df,cat_columns)
    visualize_movies_genres(df,cat_columns)

if __name__ == '__main__':
    main()
