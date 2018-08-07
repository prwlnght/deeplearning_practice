'''
Trying out how well an RNN does on data from GRUSI

does:
0. make sure this works out of the box on mnist data

todo:

1. convert the rnn example to use data from GRUSI pickles
    0. base conversion with some accuracy > 4 percent ? like random guessing
        0. run with more epochs > 12 percent train accuracy (on test_data)
        1. shuffle the data before pickling it > 12 percent
        2. try on train on train_data with batch_size > run1 >> 9 percent
        3. try on 500 length rnn cells > run1 > 13 percent
            1. also getting train_accuracy > run2 > canceled
        4. with smaller batch size of 5 > run3 >> 7 percent 
            0. batch size smaller with 10 epochs >> run 2 > canceled
    1. clock time
    2. make smaller images (and write a test script to execute different architectures assuming a pickle)
    3. get n_classes etc. dynamically
2. deploy this to the new server
3. read the 'RNN' impl paper and run through impl details on tf



'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

import pickle
import numpy as np
import datetime as dt
import platform
import os

if platform.system() == 'Windows':
    import resources_windows as resources
else:
    import resources_unix as resources

# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

hm_epochs = 5
batch_size = 16
chunk_size = 28
n_chunks = 28
rnn_size = 500

n_classes = 20  # get this dynamically

# booleans
to_train_network = True
to_test_network = False
to_log_results = True
to_print_train_accuracy = True

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

tmp_dir = os.path.join(resources.workspace_dir, 'tmp')
log_file = os.path.join(tmp_dir, 'grusi', 'run_logs.log')


def rnn_model(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1, 0, 2])  # what does this do?
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)  # what is this
    # init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output


def train_neural_network(x):
    prediction = rnn_model(x)
    # OLD VERSION:
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    start_t = dt.datetime.now()
    with open('tmp/grusi/train_data.pickle', 'rb') as f:
        train_x, train_y = pickle.load(f)

    with open('tmp/grusi/test_data.pickle', 'rb') as f2:
        test_x, test_y = pickle.load(f2)
    # with open('tmp/grusi/train_data.pickle', 'rb')

    with tf.Session() as sess:
        # OLD:
        # sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())
        print('Training a simple RNN of size: ', rnn_size, 'for :', hm_epochs, " epochs and chunk size: ", chunk_size)
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(len(train_x) / batch_size)):
                epoch_x = np.array(train_x[0:batch_size])
                epoch_y = np.array(train_y[0:batch_size])
                # reshape epoch_x
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        end_t = dt.datetime.now()
        print('Total time for training: ', end_t - start_t)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        test_x = np.array(train_x)
        test_y = np.array(train_y)
        tn_accuracy = accuracy.eval({x: test_x.reshape(-1, n_chunks, chunk_size), y: test_y})
        print('Train Accuracy:', tn_accuracy)

        test_x = np.array(test_x)
        test_y = np.array(test_y)
        test_accuracy = accuracy.eval({x: test_x.reshape(-1, n_chunks, chunk_size), y: test_y})
        print('User Independent Test Accuracy:', test_accuracy)

        with open(log_file, 'a') as m_log:
            m_log.write('-------------------------------------------------------------------------')
            m_log.write(str(dt.datetime.now()) + '\n')
            m_log.write(
                'Trained ' + str(os.path.basename(__file__)) + 'for: ' + str(end_t - start_t) + 'n_epochs: ' + str(
                    hm_epochs) + 'rnn_size: ' + str(
                    rnn_size) + 'chunk_size: ' + str(chunk_size) + '\n')
            m_log.write('train_accuracy: ' + str(tn_accuracy) + '\n')
            m_log.write(str(os.path.basename(__file__)) + ' test_accuracy: ' + str(test_accuracy) + ' \n')
            m_log.write('-------------------------------------------------------------------------')




if __name__ == '__main__':
    train_neural_network(x)
