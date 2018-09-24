import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import load_model
import os


class Model():
    def __init__(self):
        model = Sequential()
        self.model = model

    def createModel(self, embedding_vector_length, top_words, max_review_length):
        embedding_vector_length = 32
        self.model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
        self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(LSTM(100))
        self.model.add(Dense(3, activation='sigmoid'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=1, batch_size=64)

    def evaluate(self, X_test, y_test):
        # Final evaluation of the model
        scores = self.model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))

    def saveModel(self, name):
        self.model.save(name+".h5")


