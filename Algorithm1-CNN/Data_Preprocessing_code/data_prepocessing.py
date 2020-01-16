import pandas as pd
import numpy as np
import json

def print_full(input_data):
    pd.set_option('display.max_rows', len(input_data))
    pd.set_option('display.max_columns', None)
    print(input_data)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')

# original csv [45466 rows x 24 columns]
# new csv after dropping empty poster url [45080 rows x 24 columns]
# new csv after dropping empty genre [42838 rows x 24 columns]
# new csv after dropping incorrect genre [42835 rows x 24 columns]
# new csv after dropping missing imdb_id [42825 rows x 24 columns]
# 42795 distinct imdb_id [42795 rows x 23 columns]
# 42782 poster downloaded
def add_fullpath(input_data):
    main_url = 'https://image.tmdb.org/t/p/original'
    input_data['poster_path'] = main_url+ input_data['poster_path'].astype(str)
    return input_data
def get_genre(row_genre):
    row_genre = str(row_genre).replace("'", '"')
    list_genre = []
    dict_row_genre = json.loads(row_genre)
    for item in dict_row_genre:
        list_genre.append(item['name'])
    return list_genre
def process_genre(input_data):
    input_data['genres'] = input_data['genres'].apply(get_genre)
    return input_data
def get_distinct_genre(input_data):
    list_genre = []
    for index,row in input_data.iterrows():
        list_genre = list_genre+row['genres']
        # list_genre.append(str(row['imdb_id']))
    # return list_genre
    return set(list_genre)
def assign_genre(input_data):
    for index,row in input_data.iterrows():
        for each_genre in row['genres']:
            input_data.loc[index,each_genre] = 1
    return input_data
def save_df(input_data):
    file_name = 'clean_data_with_path.csv'
    input_data.to_csv(file_name, encoding='utf-8', index=False)

def main():
    input_data = pd.read_csv('original_data/movies_metadata.csv',low_memory=False)    # reading the csv file

    ### Drop rows that do not have Poster URL path
    input_data['poster_path'].replace('', np.nan, inplace=True)
    input_data.dropna(subset=['poster_path'], inplace=True)

    ### Add full poster path name (original: https://image.tmdb.org/t/p/original)
    input_data = add_fullpath(input_data)

    ### Drop rows that do not have Genre
    input_data['genres'].replace('[]', np.nan, inplace=True)
    input_data.dropna(subset=['genres'], inplace=True)
    ### Drop rows that do not have imdb_id
    input_data['imdb_id'].replace('', np.nan, inplace=True)
    input_data.dropna(subset=['imdb_id'], inplace=True)
    ### Drop rows that have duplicate imdb_id
    input_data.drop_duplicates(subset=['imdb_id'],inplace=True)
    ### Drop rows that have incorrect Genre
    input_data['title'].replace('', np.nan, inplace=True)
    input_data.dropna(subset=['title'], inplace=True)

    ### Format Genre column into readable lists
    input_data = process_genre(input_data)

    ### Create New DataFrame out of Existing one
    # new_data = input_data[['imdb_id', 'genres']].copy()
    new_data = input_data[['imdb_id','poster_path', 'genres']].copy()

    new_data = new_data.reset_index()
    new_data = new_data.reindex(index=range(0,len(input_data)))

    ### Get Genre list
    genre_list = get_distinct_genre(new_data)
    # print(genre_list)
    # print(len(genre_list))

    ### Create new DataFrame consiting of genre filled with 0
    temp_df = pd.DataFrame(0, index=np.arange(len(new_data)), columns=genre_list)

    ### Merge 2 dataframes
    new_df = new_data.join(temp_df)

    ### Drop unused index column from dataframes
    new_df = new_df.drop('index',1)

    ### Assign genre label in every row
    new_df = assign_genre(new_df)
    # print(input_data)
    # print_full(input_data[:1])

    print(new_df)

    ### Save csv files
    save_df(new_df)
if __name__ == '__main__':
    main()
