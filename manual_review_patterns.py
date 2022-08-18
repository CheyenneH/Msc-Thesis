import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


filename = 'all_labeled_data.csv'
df = pd.read_csv(filename, delimiter=';', encoding='utf-8')
k = 0
y_pred = []
for index, row in df.iterrows():
    k = k + 1
    if k < 500:
        continue
    string = row['txt'].lower()
    # find word "roken"
    smoking_index = string.find(" roken")

    if smoking_index == -1:
        y_pred.append("rookt")
        continue
    #get 40 positions before and 50 positions after #this is something I can experiment with
    start = smoking_index - 100
    end = smoking_index + 100
    new_string = string[start:end]
    print(new_string)
    print(row['label'])
    new_label = input("Please enter something: ")