"""
This program takes two files 'pos.txt' and 'neg.txt' and builds a lexicon out of them

Then it builds a feature map by counting usage of each of the lexicon words for each
The output is one hot
it then saves it to a pickle


"""

import os
import pickle
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import numpy as np
import random


data_dir = 'data\\custom0'
pos = os.path.join(data_dir, 'pos.txt')
neg = os.path.join(data_dir, 'neg.txt')
pickle_f = os.path.join(data_dir, 'sentiment_set_self.pickle')

hm_lines = 10000
lemmatizer = WordNetLemmatizer()
test_partition = 0.1

def build_lexicon(pos, neg):
    lexicon = []
    for m_file in [pos, neg]:
        #read each line
        with open(m_file, 'r') as f:
            file_contents = f.readlines()
            for line in  file_contents[:hm_lines]:
                #print(line)
                all_words = word_tokenize(line.lower())
                lexicon += list(all_words)
                # print(len(all_words))
            # print('number of lines was %d in file %s' %(len(file_contents), m_file))
        # print('lexicon length before lemmatizing', len(lexicon))
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    print('lexicon length after lemmatizing', len(lexicon))
    w_counts = Counter(lexicon)
    print('Unique words found were %d' %(len(w_counts.keys())))
    #remove any word that appears more than 1000 times and less than 50 times
    terse_lexicon = []
    for unique_word in w_counts.keys():
        if 1000 > w_counts[unique_word] > 10:
            terse_lexicon.append(unique_word)
    print('new length of lexicon', len(terse_lexicon))

    return terse_lexicon

'''
for each file 
'''
def build_features (sample, lexicon, label):
    feature_set = []
    with open(sample, 'r') as f:
        contents = f.readlines()
        for m_line in contents[:hm_lines]:
            this_features = np.zeros(len(lexicon))
            # print('the first line is', m_line)
            this_words = word_tokenize(m_line.lower())
            this_words = [lemmatizer.lemmatize(i) for i in this_words]
            for word in this_words:
                if word.lower() in lexicon:
                    this_features[lexicon.index(word.lower())] += 1
            this_features = list(this_features)
            feature_set.append([this_features, label])
    print(feature_set[0])
    return feature_set


def create_features_and_labels(pos, neg):
    print(pos, neg)
    m_lexicon = build_lexicon(pos, neg)
    # print(m_lexicon)
    all_features = []
    all_features += build_features(pos, m_lexicon, [1,0])
    all_features += build_features(neg, m_lexicon, [0,1])
    random.shuffle(all_features)
    all_features = np.array(all_features)

    test_size = int(len(all_features)*test_partition)
    print('Using %d for training and %d for testing' %(len(all_features)-test_size, test_size))
    train_x = list(all_features[:, 0][:-test_size])
    train_y = list(all_features[:, 1][:-test_size])
    test_x = list(all_features[:, 0][-test_size:])
    test_y = list(all_features[:, 1][-test_size:])

    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = create_features_and_labels(pos, neg)
    with open(pickle_f, 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
