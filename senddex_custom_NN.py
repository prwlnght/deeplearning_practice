import tensorflow as tf

'''
input > weight > hidden layer (activation f) > weights > hidden l2 > ... repeat 
for all layers 

compare the output to input > cost function (cross entropy) 
optimizer to adjust the gradients  

'''

#from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("tmp/data/", one_hot=True)  # one component is on and rest is off
import numpy as np
import tensorflow as tf
from sentdex_custom_data import create_feature_set_and_labels
import pickle


with open('data\\custom0\\sentiment_set_self.pickle', 'rb') as f:
    train_x, train_y, test_x, test_y = pickle.load(f)

'''
0 = [1, 0, 0 , 0 , 0, ... ] 
1 = [0, 1, 0, ... ] 

'''

# the three hidden layers
n_nodes_hl1 = 512
n_nodes_hl2 = 256
n_nodes_hl3 = 128

n_classes = 2
batch_size = 128  # batches of 100 examples to manipulate the weights

# use as placeholders for varied input
x = tf.placeholder('float')  # flatten the images into a vector map
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]), trainable=False)}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]), trainable=False)}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]), trainable=False)}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]), trainable=False)}

    l1 = tf.matmul(data, hidden_1_layer['weights']) + hidden_1_layer['biases']
    l1 = tf.nn.relu(l1)

    l2 = tf.matmul(l1, hidden_2_layer['weights']) + hidden_2_layer['biases']
    l2 = tf.nn.relu(l2)

    l3 = tf.matmul(l2, hidden_3_layer['weights']) + hidden_3_layer['biases']
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)  # this runs through one pass, I am guessing
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    #                         learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 25

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i  < len(train_x):
                #take batches
                start = i
                end = i+ batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))  # to get float accuracy

        print('Accuracy', accuracy.eval({x:test_x, y: test_y}))  # get test accuracy
        print('Train Accuracy', accuracy.eval({x: train_x, y: train_y}))


train_neural_network(x)
