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
import os
import pandas as pd

lemmatizer = WordNetLemmatizer()
hm_lines = 100000
to_preprocess = False
to_create_lexicon = False
to_create_test_feature_sets = False
to_init_params = False
to_shuffle_train_data = True

# data_dir = 'data\\custom0'
temp_dir = 'tmp'
tf_log = os.path.join('tmp', 'tf.log')
this_data_dir = os.path.join('data', 'custom0')
lexicon_path = os.path.join(temp_dir, 'lexicon.pickle')


# for buffering and stuff
def init_process(fin, fout):
    outfile = open(fout, 'a', encoding='latin-1')  # open to append
    skipped = 0
    with open(fin, buffering=200000, encoding='latin-1') as f:
        try:
            for line in f:
                line = line.replace('"', '')
                initial_polarity = line.split(',')[0]
                if initial_polarity == '0':
                    initial_polarity = [1, 0]
                elif initial_polarity == '4':
                    initial_polarity = [0, 1]
                else:
                    skipped += 1
                    continue
                tweet = line.split(',')[-1]
                outline = str(initial_polarity) + ':::' + tweet
                outfile.write(outline)
        except Exception as e:
            print(str(e))
    print('Skipped ', skipped, 'lines')
    outfile.close()


def create_lexicon(fin):
    lexicon = []
    with open(fin, 'r', buffering=100000, encoding='latin-1') as f:
        try:
            counter = 1
            content = ''
            for line in f:
                counter += 1
                if (counter / 2500.0).is_integer():
                    tweet = line.split(':::')[1]
                    content += ' ' + tweet
                    words = word_tokenize(content)
                    words = [lemmatizer.lemmatize(i) for i in words]
                    lexicon = list(set(lexicon + words))
                    print(counter, len(lexicon))

        except Exception as e:
            print(str(e))

    with open(os.path.join(temp_dir, 'lexicon.pickle'), 'wb') as f:
        pickle.dump(lexicon, f)


def init_data_params(f_training, f_params):
    with open(f_params, 'wb') as f:
        try:
            len_lexicon = 0
            counter = 0
            with open(f_training, 'r', encoding='latin-1') as f2:
                for line in f2:
                    # tweet = line.split(':::')
                    counter += 1
            print('length of training', counter)

        except Exception as e:
            print(str(e))
        pickle.dump([counter], f)


def save_as_pickle(m_this_test_file, save_pickle_path, m_lexicon_path=lexicon_path):
    test_data = []
    counter = 0
    features_sets = []
    labels = []
    with open(m_lexicon_path, 'rb') as f:
        lexicon = pickle.load(f)
    with open(m_this_test_file, buffering=20000) as f:
        for line in f:
            try:
                label = list(eval(line.split(':::')[0]))
                tweet = line.split(':::')[1]
                current_words = word_tokenize(tweet.lower())
                current_words = [lemmatizer.lemmatize(i) for i in current_words]

                features = np.zeros(len(lexicon))

                for word in current_words:
                    if word.lower() in lexicon:
                        index_value = lexicon.index(word.lower())
                        features[index_value] += 1
                features_sets.append(list(features))
                labels.append(label)
                counter += 1
            except Exception as e:
                print(str(e))
    print('Test data size is', counter, 'samples')
    with open(save_pickle_path, 'wb') as f3:
        pickle.dump([features_sets, labels], f3)

def shuffle_data(fin):
    df = pd.read_csv(fin, error_bad_lines=False, encoding='latin-1')
    df = df.iloc[np.random.permutation(len(df))]
    print(df.head())
    df.to_csv(os.path.join(this_data_dir, 'train_set_shuffled.csv'), index=False)



if __name__ == '__main__':

    if to_preprocess:
        init_process(os.path.join(this_data_dir, 'training.1600000.processed.noemoticon.csv'),
                     os.path.join(this_data_dir, 'train_set.csv'))
        init_process(os.path.join(this_data_dir, 'testdata.manual.2009.06.14.csv'),
                     os.path.join(this_data_dir, 'test_set.csv'))

    if to_init_params:
        init_data_params(os.path.join(this_data_dir, 'train_set.csv'), os.path.join(temp_dir, 'parameters.txt'))

    if to_create_lexicon:
        create_lexicon(os.path.join(this_data_dir, 'train_set.csv'))

    if to_create_test_feature_sets:
        test_file = os.path.join(this_data_dir, 'test_set.csv')
        if not os.path.exists(test_file):
            print('Test file path does not exist, please check resources file')
            exit()
        if not os.path.exists(lexicon_path):
            print('lexicon does not exist')
            exit()
        test_file_preprocessed = os.path.join(this_data_dir, 'test_file_processed.pickle')
        save_as_pickle(test_file, test_file_preprocessed, lexicon_path)

    if to_shuffle_train_data:
            shuffle_data(os.path.join(this_data_dir, 'train_set.csv'))
