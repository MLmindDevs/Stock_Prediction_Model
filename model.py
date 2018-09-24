import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from data_prepro import extract_data
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import load_model
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) # Tensorflow gpu support

# use the function from the data_prepro file
labels, news, dates = extract_data()
print(len(news), len(labels))
labels = to_categorical(labels, num_classes=3)
# split our data into a training and a testing set.
X_train, X_test, y_train, y_test = train_test_split(news, labels, test_size=0.2, random_state=45)
top_words = 120000
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=1, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
model.save("model_v0.1.h5")