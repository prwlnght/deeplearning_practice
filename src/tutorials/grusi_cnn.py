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

# which_model_to_run = 'cnn'
which_model_to_run = 'cnn'

hm_epochs = 4000
batch_size = 16
chunk_size = 500  # this is dynamically set from data_size assuming square images
n_chunks = 500  # this is dynamically set from data size assuming square images
rnn_size = 128
image_height = 500
image_width = 500
n_classes = 20  # get this dynamically

# booleans
to_train_network = True
to_test_network = False
to_log_results = True
to_print_train_accuracy = True
to_get_globals_from_test_file = True
to_save_models = False
to_load_previous_models = False
to_validate = True
to_clear_previous_models = True
is_color_images = True

if is_color_images:
    channels = 3  # change this
else:
    channels = 1

x = tf.placeholder('float', [None, image_width, image_height])
y = tf.placeholder('float')
global_step = tf.train.get_or_create_global_step()

if which_model_to_run == 'cnn':
    models_dir = os.path.join(tmp_dir, 'models_cnn')
else:
    models_dir = os.path.join(tmp_dir, 'models_rnn')
if not os.path.exists(models_dir):
    os.mkdir(models_dir)

pickles_path = os.path.join(tmp_dir, '100_pickles')
train_pickle_name = 'train_data.pickle'
test_pickle_name = 'test_data.pickle'

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def cnn_model(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_conv3': tf.Variable(tf.random_normal([5, 5, 64, 132])),
               # 'W_fc': tf.Variable(tf.random_normal([8 * 8 * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_conv3': tf.Variable(tf.random_normal([132])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, image_width, image_height, channels])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
    conv3 = maxpool2d(conv3)

    fc_shape = int(conv3.get_shape().as_list()[1] * conv3.get_shape().as_list()[2] * conv3.get_shape().as_list()[3])
    weights['W_fc'] = tf.Variable(tf.random_normal([fc_shape, 1024]))

    fc = tf.reshape(conv3, [-1, fc_shape])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


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


def get_globals_from_test_file():
    global n_chunks
    global chunk_size
    global image_height
    global image_width
    global n_classes
    with open(os.path.join(pickles_path, test_pickle_name), 'rb') as f2:
        test_x, test_y = pickle.load(f2)
        n_chunks = int(math.sqrt(len(test_x[0])))
        chunk_size = n_chunks
        n_classes = len(test_y[0])
        if is_color_images:
            image_width = int(math.sqrt(len(test_x[0]) / 3)) * 3
            image_height = int(math.sqrt(len(test_x[0]) / 3))
        else:
            image_height = chunk_size
            image_width = image_height


def train_neural_network(x):
    if which_model_to_run == 'cnn':
        prediction = cnn_model(x)
    else:
        prediction = rnn_model(x)
        pass
    # OLD VERSION:
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost, global_step=global_step)

    start_t = dt.datetime.now()

    with open(os.path.join(pickles_path, test_pickle_name), 'rb') as f2:
        test_x, test_y = pickle.load(f2)
    # with open('tmp/grusi/train_data.pickle', 'rb')

    train_pickles = os.listdir(pickles_path)
    train_pickles = [this_name for this_name in train_pickles if this_name.startswith('train')]

    train_pickle_file_counter = 0
    with open(os.path.join(pickles_path, train_pickles[train_pickle_file_counter]), 'rb') as f:
        train_x, train_y = pickle.load(f)

    # tf.reset_default_graph()

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # print(sess.run(v1))
        # save_path = saver.save(sess, './model.ckpt')
        # print("model saved in file:", save_path)
        # # Create an op to increment v1, run it, and print the result.
        #
        # sess.run(increment_op)
        # print(sess.run(v1))
        #
        # # Restore from the checkpoint saved above.
        # saver.restore(sess, './model.ckpt')
        # print(sess.run(v1))

        if which_model_to_run == 'cnn':
            print('Training a simple CNN of size 3 for :', hm_epochs, " epochs and image size: ", image_width,
                  'with batch size: ', batch_size)
        else:
            print('Training a simple RNN of size: ', rnn_size, 'for :', hm_epochs, " epochs and chunk size: ",
                  chunk_size,
                  'with batch size: ', batch_size)

        if to_load_previous_models:
            try:
                saved_models_list = sorted(list(map(int, os.listdir(models_dir))))
                epoch = int(saved_models_list[-1])
                this_models_dir = os.path.join(models_dir, str(saved_models_list[-1]))
                model_file = os.path.join(this_models_dir, 'm_model.ckpt')
                saver.restore(sess, model_file)
                # print(sess.run(v2))
                model_validation_x = np.array(test_x)
                model_validation_y = np.array(test_y)
                tn_accuracy = accuracy.eval(
                    {x: model_validation_x.reshape(-1, image_height, image_width), y: model_validation_y})
                print('The previous model at epoch', epoch, 'had a test accuracy of: ', tn_accuracy)

                print('starting at epoch: ', epoch)
            except Exception as e:
                print('No previous training file found ..............')
                # sess.run(tf.global_variables_initializer())
                epoch = 0
        else:
            epoch = 0

        while epoch < hm_epochs:
            # saved_epoch =tf.train.global_step(sess, global_step)
            epoch += 1
            # if epoch isn't 1, load the model

            if epoch != 1:
                if epoch % 10 == 0:
                    model_validation_x = np.array(test_x)
                    model_validation_y = np.array(test_y)
                    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                    tn_accuracy = accuracy.eval(
                        {x: model_validation_x.reshape(-1, image_width, image_height), y: model_validation_y})
                    print('The model at epoch', epoch, 'had a test accuracy of: ', tn_accuracy)

            epoch_loss = 0
            batch_start_marker = 0
            batch_end_marker = 0

            while batch_end_marker < len(train_x):
                batch_end_marker += batch_size
                if batch_end_marker > len(train_x):
                    # if more training files to load > load > else set as below
                    train_pickle_file_counter += 1
                    if train_pickle_file_counter < len(train_pickles):
                        with open(os.path.join(pickles_path, train_pickles[train_pickle_file_counter]), 'rb') as f:
                            x_, y_ = pickle.load(f)
                            x_ += train_x[batch_start_marker: len(train_x) - 1]
                            y_ += train_y[batch_start_marker:len(train_y) - 1]
                            train_x = x_
                            train_y = y_
                            batch_start_marker = 0
                            batch_end_marker = batch_size
                            if batch_end_marker > len(train_x):
                                batch_end_marker = len(train_x) - 1
                    else:
                        batch_end_marker = len(train_x)

                if batch_end_marker > batch_start_marker:
                    batch_x = np.array(train_x[batch_start_marker:batch_end_marker])
                    batch_y = np.array(train_y[batch_start_marker:batch_end_marker])
                    # reshape epoch_x
                    m_batch_size = len(batch_x)
                    batch_x = batch_x.reshape((m_batch_size, image_width, image_height))

                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                    epoch_loss += c
                    batch_start_marker = batch_end_marker

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            # save model
            if to_save_models:
                this_models_dir = os.path.join(models_dir, str(epoch))
                if not os.path.exists(this_models_dir):
                    os.mkdir(this_models_dir)

                saver.save(sess, os.path.join(this_models_dir, 'm_model.ckpt'))
        end_t = dt.datetime.now()
        print('Total time for training: ', str(end_t - start_t), 'for :', int(len(train_x) / batch_size), 'batches')

        validate_x = np.array(train_x)
        validate_y = np.array(train_y)
        tn_accuracy = accuracy.eval({x: validate_x.reshape(-1, image_width, image_height), y: validate_y})
        print('Train Accuracy:', tn_accuracy)

        test_x = np.array(test_x)
        test_y = np.array(test_y)
        test_accuracy = accuracy.eval({x: test_x.reshape(-1, image_width, image_height), y: test_y})
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
            if to_save_models:
                m_log.write('Latest model saved as:' + str(this_models_dir))
            m_log.write('-------------------------------------------------------------------------')


if __name__ == '__main__':
    if to_get_globals_from_test_file:
        get_globals_from_test_file()
        x = tf.placeholder('float', [None, image_width, image_height])
        y = tf.placeholder('float')
    if to_clear_previous_models:
        try:
            shutil.rmtree(models_dir)
            if not os.path.exists(models_dir):
                os.mkdir(models_dir)
        except IOError as e:
            print(str(e))
    v1 = tf.Variable(tf.zeros([2, 2], dtype=tf.float32, name='v1'))
    train_neural_network(x)
