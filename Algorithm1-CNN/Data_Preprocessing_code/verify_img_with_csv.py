import pandas as pd
import os
''' This code helps verify that the csv data is matched with the poster movie images
    we have uploaded since some poster link does not work. 
    The data which do not have their own movie poster will be removed. ''' 

def remove_error_poster(input_data,list_poster):
    for index,row in input_data.iterrows():
        img_name = row['imdb_id']+'.jpg'
        if img_name in list_poster:
            continue
        else:
            input_data = input_data.drop([index])#.reset_index(drop=True)
    return input_data
def save_df(input_data):
    file_name = 'new_clean_data_with_path.csv'
    input_data.to_csv(file_name, encoding='utf-8', index=False)

def main():
    input_data = pd.read_csv('clean_data_with_path.csv',low_memory=False)    # reading the csv file
    list_current_poster_file = os.listdir('poster_images')

    new_input = remove_error_poster(input_data,list_current_poster_file)
    print(new_input)
    save_df(new_input)
if __name__ == '__main__':
    main()
