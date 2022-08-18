from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Conv1D, MaxPooling1D, Concatenate
from tensorflow.keras import Sequential
import pandas as pd
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.utils.vis_utils import plot_model
import graphviz

"""
#read labels
df=pd.read_csv('labels.csv', sep=',',header=None, skiprows=1)
df = df.replace({'rookt': 1, 'voorheen': 0})
labels = df.to_numpy()
print(len(labels))
print('Datatype:', labels.dtype)

#opening the file again!
all_embeddings = []
text_file = open("foo.csv", "r")
lines = text_file.readlines()
longest_list = 0
total_length = 0
for line in lines:
    line = line.replace('[', '')
    line = line.replace(']', '')
    line = line.replace(',', '')
    embedding = line.split(" ")
    length = len(embedding)
    total_length += length
    if length > longest_list:
        longest_list = length
    all_embeddings.append(embedding)

print(longest_list)
print(total_length/len(all_embeddings))

text_file.close()
print(len(all_embeddings))

#convert list to numpy
numpy_embedding = np.array(all_embeddings, dtype=object)
print(numpy_embedding.shape)

#padding
#l = np.array([len(numpy_embedding[i]) for i in range(len(numpy_embedding))])
width = 30000
b=[]
for i in range(len(numpy_embedding)):
    if len(numpy_embedding[i]) <= width:
        x = np.pad(numpy_embedding[i], (0,width-len(numpy_embedding[i])), 'constant',constant_values = 0)
    elif len(numpy_embedding[i]) >= width:
        x = numpy_embedding[i][:width]
    else:
        x = numpy_embedding[i]
    b.append(x)
b = np.array(b)
b = b.astype(float)
print(b.shape)
print('Datatype:', b.dtype)

small_test_x = b[:100]
small_test_x = small_test_x.reshape((small_test_x.shape[0], small_test_x.shape[1], 1))
small_test_y = labels[:100]
print(small_test_x.shape, len(small_test_y))

data = b.reshape((b.shape[0], b.shape[1], 1))
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
"""

#do the RNN magic
#we can skip the embedding layer because we already have our own
#dropout for regularization
model = Sequential() #
model.add(LSTM(32, input_shape=(94000, 1)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
"""
model.fit(X_train, y_train, epochs=1, batch_size=64)
#resulted in a final training accuracy of 0.58

#now for the test set on model
y_pred = model.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))
"""


model1 = Sequential() #
model1.add(LSTM(64, input_shape=(94000, 1)))
model1.add(Dropout(rate=0.2))
model1.add(Dense(64, activation='relu'))
model1.add(Dropout(rate=0.2))
model1.add(Dense(1, activation='softmax'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model1.summary())
plot_model(model1, to_file='model1_plot.png', show_shapes=True, show_layer_names=True)
"""
model1.fit(X_train, y_train,epochs=1, batch_size=64)
#resulted in a final training accuracy of 0.58

#now for the test set on model 1
y_pred = model1.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))

"""
model3 = Sequential()
model3.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu', input_shape=(94000, 1)))
model3.add(MaxPooling1D(pool_size=4))
model3.add(LSTM(64, return_sequences=True))
model3.add(Dropout(rate=0.2))
model3.add(Dense(1, activation='softmax'))
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model3.summary())
plot_model(model3, to_file='model_plot3.png', show_shapes=True, show_layer_names=True)
"""
model3.fit(X_train, y_train, epochs=1, batch_size=64)
#resulted in a final training accuracy of 0.61

#now for the test set on model 1
y_pred = model3.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))
"""
model2 = Sequential() #
model2.add(LSTM(64, input_shape=(94000, 1)))
model2.add(Dropout(rate=0.5))
model2.add(Dense(1, activation='softmax'))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model2.summary())
plot_model(model2, to_file='model2_plot.png', show_shapes=True, show_layer_names=True)

"""
model2.fit(X_train, y_train, epochs=1, batch_size=64)
#resulted in a final training accuracy of 0.59

#now for the test set on model 2
y_pred = model2.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))
"""





