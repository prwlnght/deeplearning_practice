'''

Sub-project: EAI, how does RNN work? and why does RNN work?

Goals:

1. Implement a classifier based on Recurrent Neural Network and understand its compoments
2. Implement an auto-encoder to 'predict' the next time-stamp
3. Train on the entire network, then animate
4. Switch datasets to dance


Input:
x,y coodinate data for a bunch of bone locations > learn2Sign

class_counter = {}
Ouput:
Varies by step


this file goal:
1. Implement a classifier based RNN
    1.1. Import data from the .csv and class label from the filename/foldername
    1.2. Build a classifier based on a RNN with some structure then train and test
    1.3. Switch the classifier RNN to an LSTM, then train and test.


'''

import os, platform, sys
import csv
import pandas
data_dir = '/Users/hayden/workspace/learn2Sign/data/tmp/csvs/video_input/'
import itertools
import numpy as np
import pickle


#booleans
to_write_pickles = False

def write_to_pickle():
    # get all classes
    for this_class in os.listdir(data_dir):
        if not this_class.startswith('.'):
            if not all_classes.__contains__(this_class):
                all_classes.append(this_class)

    print(all_classes)
    for this_class in os.listdir(data_dir):
        if not this_class.startswith('.'):
            if not all_classes.__contains__(this_class):
                all_classes.append(this_class)
            for m_sample in os.listdir(os.path.join(data_dir, this_class)):
                if m_sample.endswith('.csv'):
                    df = pandas.read_csv(os.path.join(data_dir, this_class, m_sample))
                    this_x = np.array([df['rightWrist_x'], df['rightWrist_y'], df['leftWrist_x'], df['rightWrist_y']])
                    this_y = np.zeros(len(all_classes))
                    this_y[all_classes.index(this_class)] += 1
                    this_features = [this_x, this_y]
                    all_features.append(this_features)

    with open('all_data.pkl', 'wb') as f3:
        pickle.dump(all_features, f3)


if __name__ == '__main__':
    class_counter = {}
    all_classes = []
    all_features = []

    if to_write_pickles:
        write_to_pickle()

    #start rnn code
    '''
    todo:
    construct an rnn that at each timestep takes as input x, y cooridates for each of the significant bones
    how to take into account the 'score'? next step 
    
    '''

