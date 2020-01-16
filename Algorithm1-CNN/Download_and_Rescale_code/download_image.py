import urllib.request as req
import pandas as pd
from threading import Thread
import os
import requests
import shutil
''' Code for downloading original poster movies from url links '''
input_data = pd.read_csv('clean_data_with_path.csv',low_memory=False)    # reading the csv file
list_current_poster_file = os.listdir('poster_images')

def is_aldy_exist(imdb_id):
    img_name = imdb_id + ".jpg"
    if img_name in list_current_poster_file:
        print("File: ", img_name, " already downloaded!")
        return True
    return False
def load_image(input_data):
    for index,row in input_data.iterrows():
        img_url = row['poster_path']
        img_name = row['imdb_id']
        if is_aldy_exist(img_name):
            continue

        # req.urlretrieve(img_url, "poster_images/"+img_name+".jpg")
        r = requests.get(img_url, stream=True)
        if r.status_code == 200:
            img_name_jpg= "poster_images/"+img_name+".jpg"
            with open(img_name_jpg, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)

        print(index,":Download ",img_name," successful")

def main1():
    load_image(input_data[:8000])
def main2():
    load_image(input_data[8000:16000])
def main3():
    load_image(input_data[16000:24000])
def main4():
    load_image(input_data[24000:32000])
def main5():
    load_image(input_data[32000:])

def main():
    print(list_poster_file)

if __name__ == '__main__':
    # main()
    Thread(target = main1).start()
    Thread(target = main2).start()
    Thread(target = main3).start()
    Thread(target = main4).start()
    Thread(target = main5).start()
