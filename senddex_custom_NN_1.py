import tensorflow as tf

'''
input > weight > hidden layer (activation f) > weights > hidden l2 > ... repeat 
for all layers 

compare the output to input > cost function (cross entropy) 
optimizer to adjust the gradients  

'''

# from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("tmp/data/", one_hot=True)  # one component is on and rest is off
import numpy as np
import tensorflow as tf
from sentdex_custom_data import create_feature_set_and_labels
import pickle

with open('data\\custom0\\sentiment_set.pickle', 'rb') as f:
    train_x, train_y, test_x, test_y = pickle.load(f)

'''
0 = [1, 0, 0 , 0 , 0, ... ] 
1 = [0, 1, 0, ... ] 

'''

# define x and y (as input to the NN)

x = tf.placeholder('float', (None, len(train_x[0])))
y = tf.placeholder('float', (None, len(train_y[0])))

n_nodes_hl1 = 516
n_nodes_hl2 = 256
n_nodes_hl3 = 128

n_classes = 2
hm_epochs = 25
batch_size = 128


def NN_model(data):
    hidden_l1_layer = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                       'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_l2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                       'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_l3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                       'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.matmul(data, hidden_l1_layer['weights']) + hidden_l1_layer['biases']
    l2 = tf.matmul(l1, hidden_l2_layer['weights']) + hidden_l2_layer['biases']
    l3 = tf.matmul(l2, hidden_l3_layer['weights']) + hidden_l3_layer['biases']

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train_NN(x):
    prediction = NN_model(x)

    # define a cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        # cycle through for epohs
        sess.run(tf.global_variables_initializer())
        for epoch_counter in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while (i < len(train_x)):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c / batch_size
                i += batch_size
            print('Loss at Epoch %d was %f' % (epoch_counter, epoch_loss))

        correct = tf.equal(tf.arg_max(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('train_accuracy', accuracy.eval({x: train_x, y: train_y}))
        print('test_accuracy', accuracy.eval({x: test_x, y: test_y}))


train_NN(x)
