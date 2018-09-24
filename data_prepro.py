import pandas as pd 
from collections import Counter
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def extract_data():
    data = pd.read_csv("data\RedditNews.csv") 

    dates = data['Date'].values.tolist()  # drop the values into lists
    news = data['News'].values.tolist() 

    for i in range(len(dates)): 
        dates[i] = dates[i].split("-")
    
    analyzer = SentimentIntensityAnalyzer() # init sentiment intensity analyzer
    labels = []
    for new in news:
        vs = analyzer.polarity_scores(new) 
        labels.append(analyzeCompound(vs['compound'])) # analyze every compound for the sentiments
    
    news_processed, labels_processed = lexicon_labeling_prepro(news, labels) # create the lookup table
    return (labels_processed, news_processed, dates) 


def analyzeCompound(comp): # Function to analyze the compound
    if comp>=0.05:
        return 1
    elif comp > -0.05 and comp < 0.05:
        return 0
    else:
        return -1

def lexicon_labeling_prepro(all_news, labels): # function for error correction and lookup table transformation
    # Count total words
    word_count = Counter()
    for post in all_news:
        word_count.update(post.split(" "))

    vocab_len = len(word_count)

    print("vocabulary length: " + str(vocab_len))
    # Create a look up table 
    vocab = sorted(word_count, key=word_count.get, reverse=True)
    # Create your dictionary that maps vocab words to integers here
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    news_ints=[]
    labels_last = []
    for i in range(len(all_news)):
        try:
            news_ints.append([vocab_to_int[word] for word in all_news[i].split()])
            labels_last.append(labels[i])
        except KeyError:
            print("value error... bypassing... ")


    non_zero_idx = [ii for ii, post in enumerate(news_ints) if len(post) != 0]
    news_ints = np.array([news_ints[ii] for ii in non_zero_idx])
    labels = np.array([labels_last[ii] for ii in non_zero_idx])
    return news_ints, labels
