'''
Data: Positive and Negative
The strings are text, and the sentences are of different length, so a different length of vectors


first create a lexicon of all unique words and then bag of words for each sentence
'''

import tensorflow as tf
import numpy as np
# import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer  # different forms of a words will be reduced
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 100000


def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    print('lexicon length after lemmatizing', len(lexicon))
    w_counts = Counter(lexicon) #returns a dicionary

    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 10:
            l2.append(w)
    print('lexicon size was:', len(l2))
    return l2


def sample_handling(sample, lexicon, classification):
    feature_set = []

    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            feature_set.append([features, classification])
    return feature_set


def create_feature_set_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling(pos, lexicon, [1, 0])
    features += sample_handling(neg, lexicon, [0, 1])
    random.shuffle(features)

    features = np.array(features)
    testing_size = int(test_size * len(features))

    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])

    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_set_and_labels('data\\custom0\\pos.txt', 'data\\custom0\\neg.txt')
    with open('data\\custom0\\sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
