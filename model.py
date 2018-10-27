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
        self.model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
        self.model.add(LSTM(256))
        self.model.add(Dense(3, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())
        
    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train, epochs=2, batch_size=64, validation_data=(X_val, y_val))

    def evaluate(self, X_test, y_test):
        # Final evaluation of the model
        scores = self.model.evaluate(X_test, y_test, verbose=0)
        print(scores)
        print("Accuracy: %.2f%%" % (scores[1]*100))

    def saveModel(self, name):
        path = os.getcwd()
        os.chdir(path+"/models")
        self.model.save(name+".h5")
        os.chdir("../")
    
    def loadModel(self, filepath):
        self.model = load_model(filepath)


