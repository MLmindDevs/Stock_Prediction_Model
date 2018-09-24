# Stock_Prediction_Model
A use of sentiment analytics exploiting the vader Sentiment lexicon analyzer and some Deep learning usage to get inference and stock predictions.


## Version 0.1
### Goals
The goal of this project is to get a good stock prediction rate out of the data provided using a multi-modal (ensemble) approach. The first thoughts are
1. Model to get good sentiment predictions
2. Model that gets the output of the first model as a feature and a time-series deep learning approach on the stock values.

### Requirements
You can always install the requirements through pip
1. tensorflow 
2. keras
3. pandas

### data_prepro.py
1. exporting data from the RedditNews.csv
2. exporting dates and split them by delimiter
3. labeling through the use of the Vader Sentiment Lexicon
4. check for value errors in the data.
5. build the lookup table

### model.py
Simple RNN-CNN model with one embedding layer and a 3 node output that decides the result the labels are transformed into one hot encoded vectors.
That's an alpha version model hopefully it will begin to grow into a bigger one in the process...
The current Model is trainable on the data and gets a prediction accuracy of 94% with just 4 epochs.
