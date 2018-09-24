import numpy
from keras.preprocessing import sequence
from data_prepro import extract_data
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import load_model
import tensorflow as tf
import os
from model import Model

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) # Tensorflow gpu support

# use the function from the data_prepro file
labels, news, dates = extract_data()
labels = to_categorical(labels, num_classes=3)
# split our data into a training and a testing set.
X_train, X_test, y_train, y_test = train_test_split(news, labels, test_size=0.2, random_state=45)
top_words = 120000
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

model = Model() 
model.createModel(32, top_words, max_review_length)
model.fit(X_train, y_train)
model.evaluate(X_test, y_test)
model.saveModel("version1")