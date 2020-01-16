import numpy as np
import pandas as pd
from tqdm import tqdm


train = pd.read_csv('test.csv', low_memory=False)
y_test = train.drop(['id','imdb_id', 'title','overview','overview length','genres','labels_index'], axis = 1)
# y_test = y_test.drop('genres', 1)
y_test_list = y_test.values.tolist()
print(len(y_test_list))


preds = pd.read_csv('test_result_RF.csv', low_memory=False, index_col=False)
preds = preds.drop(preds.columns[0], axis=1)
#preds = preds.iloc[1:]
preds_df = preds.applymap(int)
preds_list = preds_df.values.tolist()
print(len(preds_list))



# Calculate the 'all genres accuracy'
count = 0
for i in range(0,len(preds_list)):
    y = y_test_list[i]
    p = preds_list[i]
    if y==p:
        count = count + 1

all_accuracy = count/len(preds_list)

print(all_accuracy)


# Need to find a new way to calculate the all-genre


# Calculate the 'at least one-genre'
one_count = 0
for i in range(0,len(preds_list)):
    y = y_test_list[i]
    p = preds_list[i]
    for j in range(0,20):
        if (y[j] == 1) & (p[j] == 1) & (y[j] == p[j]):
            one_count = one_count + 1
            break

one_accuracy = one_count/3109

print(one_accuracy)

# Calculate the '1'
times_count = 0
for i in range(0,len(preds_list)):
    p = preds_list[i]
    for j in range(0,20):
        if p[j] == 1:
            times_count = times_count + 1
times = times_count / len(preds_list)
print(times)



