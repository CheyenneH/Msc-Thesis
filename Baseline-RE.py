import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


filename = 'all_labeled_data.csv'
df = pd.read_csv(filename, delimiter=';', encoding='utf-8')
#listrange = [(10, 10), (20,20), (40,40), (60,60), (80,80), (100,100), (10,20),
#             (40,20), (20,40), (10,40), (20,80), (20,100), (60,80)]
listrange = [(80,80)]
for i in listrange:
    print(i)
    pre = i[0]
    post = i[1]
    y_pred = []
    for index, row in df.iterrows():
        string = row['txt'].lower()
        # find word "roken"
        smoking_index = string.find(" roken")

        if smoking_index == -1:
            y_pred.append("rookt")
            continue
        #get 40 positions before and 50 positions after #this is something I can experiment with
        start = smoking_index - pre
        end = smoking_index + post
        new_string = string[start:end]
        #new_string = string

        #now apply word matching
        #rules here for rookt or voorheen
        if ("roken+" in new_string) or ("roken +" in new_string):
            y_pred.append("rookt")
        elif ("door" in new_string) and ("stoppen" in new_string):
            y_pred.append("voorheen")
        elif ("gevolg" in new_string) and ("stoppen" in new_string):
            y_pred.append("voorheen")
        elif ("roken-" in new_string) or ("roken -" in new_string):
            y_pred.append("voorheen")
        elif ("tot" in new_string) and ("jaar" in new_string):
            y_pred.append("voorheen")
        elif ("gestopt" in new_string) or ("gestaakt" in new_string):
            y_pred.append("voorheen")
        elif ("stoppen" in new_string) or ("staken" in new_string) or ("minder" in new_string) or ("persisterend" in new_string):
            y_pred.append("rookt")
        elif ("per dag" in new_string) or ("packyears" in new_string) or ("/dag" in new_string) or ("pakje" in new_string) or ("pack" in new_string):
            y_pred.append("rookt")
        elif ("pd" in new_string) or ("py" in new_string):
            y_pred.append("rookt")
        else:
            y_pred.append("rookt")

        print(new_string, "\n")

    df["y_pred"] = y_pred




    #calculate accuracy
    print(classification_report(df["label"], df["y_pred"]))






"""
#testing
#result = word.find('geeks')
for index, row in df.iterrows():
    string = row['txt'].lower()
    #find word "roken"
    smoking_index = string.find(" roken")

    if smoking_index == -1:
        continue
    #print 20 posities voor en 20 posities na
    print(smoking_index)
    start = smoking_index - 40
    end = smoking_index + 50
    print(string[start:end])
    print(row['label'])
"""


"""
PATTERNS FOUND:
"roken+" = current smoker
roken + stoppen = current smoker
roken + geen + !pakjes/packyears etc = non current smoker
roken + gestopt = non current smoker

else:
current smoker

"""