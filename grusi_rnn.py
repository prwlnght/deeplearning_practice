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
import math
import shutil

if platform.system() == 'Windows':
    import resources_windows as resources

    tmp_dir = os.path.join(resources.workspace_dir, 'tmp', 'grusi')
    log_file = os.path.join(tmp_dir, 'run_logs.log')
else:
    import resources_unix as resources

    tmp_dir = os.path.join(resources.workspace_dir, 'tmp', 'grusi')
    log_file = os.path.join(tmp_dir, 'run_server_logs.log')

# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

hm_epochs = 250
batch_size = 128
chunk_size = 20  # this is dynamically set from data_size assuming square images
n_chunks = 20  # this is dynamically set from data size assuming square images
rnn_size = 500

n_classes = 20  # get this dynamically

# booleans
to_train_network = True
to_test_network = False
to_log_results = True
to_print_train_accuracy = True
to_update_chunk_size_and_batch = True
to_save_and_load_models = True
to_validate = True
to_clear_previous_models = True

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')
global_step = tf.train.get_or_create_global_step()

models_dir = os.path.join(tmp_dir, 'models')
if not os.path.exists(models_dir):
    os.mkdir(models_dir)

train_pickle_name = 'train_data.pickle'
test_pickle_name = 'test_data.pickle'


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


def update_chunk_size_and_number():
    global n_chunks
    global chunk_size
    with open(os.path.join(tmp_dir, test_pickle_name), 'rb') as f2:
        test_x, _ = pickle.load(f2)
        n_chunks = int(math.sqrt(len(test_x[0])))
        chunk_size = n_chunks


def train_neural_network(x):
    saver = tf.train.Saver(max_to_keep=50)
    prediction = rnn_model(x)
    # OLD VERSION:
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost, global_step=global_step)

    start_t = dt.datetime.now()
    with open(os.path.join(tmp_dir, train_pickle_name), 'rb') as f:
        train_x, train_y = pickle.load(f)

    with open(os.path.join(tmp_dir, test_pickle_name), 'rb') as f2:
        test_x, test_y = pickle.load(f2)
    # with open('tmp/grusi/train_data.pickle', 'rb')

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # OLD:
        # sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())
        print('Training a simple RNN of size: ', rnn_size, 'for :', hm_epochs, " epochs and chunk size: ", chunk_size,
              'with batch size: ', batch_size)

        try:
            this_models_dir = os.path.join(models_dir, str(sorted(list(map(int, os.listdir(models_dir))))[-1]))
            saver.restore(sess, os.path.join(this_models_dir, 'model.ckpt'))
            train_steps_so_far = tf.train.global_step(sess, global_step)
            epoch = int(train_steps_so_far / (int(len(train_x) / batch_size)))
            print('starting at epoch: ', epoch)
        except Exception as e:
            epoch = 1

        while epoch <= hm_epochs:
            # saved_epoch =tf.train.global_step(sess, global_step)

            # if epoch isn't 1, load the model
            if epoch != 1:
                model_validation_x = np.array(test_x)
                model_validation_y = np.array(test_y)
                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                tn_accuracy = accuracy.eval(
                    {x: model_validation_x.reshape(-1, n_chunks, chunk_size), y: model_validation_y})
                print('The previous model at epoch', epoch, 'had a test accuracy of: ', tn_accuracy)

            epoch_loss = 0
            batch_start_marker = 0
            batch_end_marker = batch_size
            for _ in range(int(len(train_x) / batch_size)):
                batch_x = np.array(train_x[batch_start_marker:batch_end_marker])
                batch_y = np.array(train_y[batch_start_marker:batch_end_marker])
                # reshape epoch_x
                batch_x = batch_x.reshape((batch_size, n_chunks, chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                batch_start_marker = batch_end_marker
                batch_end_marker += batch_size

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            # save model
            this_models_dir = os.path.join(models_dir, str(epoch))
            if not os.path.exists(this_models_dir):
                os.mkdir(this_models_dir)

            saver.save(sess, os.path.join(this_models_dir, 'model.ckpt'))
            epoch += 1
        end_t = dt.datetime.now()
        print('Total time for training: ', str(end_t - start_t), 'for :', int(len(train_x) / batch_size), 'batches')

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        validate_x = np.array(train_x)
        validate_y = np.array(train_y)
        tn_accuracy = accuracy.eval({x: validate_x.reshape(-1, n_chunks, chunk_size), y: validate_y})
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
            m_log.write('Latest model saved as:' + str(this_models_dir))
            m_log.write('-------------------------------------------------------------------------')


if __name__ == '__main__':
    if to_update_chunk_size_and_batch:
        update_chunk_size_and_number()
        x = tf.placeholder('float', [None, n_chunks, chunk_size])
        y = tf.placeholder('float')
    if to_clear_previous_models:
        try:
            shutil.rmtree(models_dir)
            if not os.path.exists(models_dir):
                os.mkdir(models_dir)
        except IOError as e:
            print(str(e))
    train_neural_network(x)
