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
from sentdex_custom_data_millions import *
import pickle

n_nodes_hl1 = 500
n_nodes_hl2 = 500

n_classes = 2

batch_size = 32
total_batches = int(1600000 / batch_size)  # how to get this in line automatically
hm_epochs = 20

x = tf.placeholder('float')
y = tf.placeholder('float')

# booleans
to_train = False
to_test = True
to_usage_testing = True

temp_dir = os.path.join('', 'tmp')

tf_log = os.path.join('tmp', 'tf.log')
this_data_dir = os.path.join('data', 'custom0')
lexicon_path = os.path.join(temp_dir, 'lexicon.pickle')

hidden_1_layer = {'f_fum': n_nodes_hl1,
                  'weight': tf.Variable(tf.random_normal([2623, n_nodes_hl1])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum': n_nodes_hl2,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum': n_classes,
                'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias': tf.Variable(tf.random_normal([n_classes]))}


# initialize the nn

def neural_network(data):
    l1 = tf.matmul(data, hidden_1_layer['weight']) + hidden_1_layer['bias']
    l1 = tf.nn.relu(l1)

    l2 = tf.matmul(l1, hidden_2_layer['weight']) + hidden_2_layer['bias']
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']

    return output


# that is how we saved the model at checkpoint (todo set this up into a 'temp' folder)

'''Training the network'''


def train_neural_network(x):
    saver = tf.train.Saver(max_to_keep=50)
    prediction = neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    models_dir = os.path.join(temp_dir, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            epoch = int(open(tf_log, 'r').read().split('\n')[-2]) + 1
            print('Starting:', epoch)
        except:
            epoch = 1  # if logfile isn't there, then restart training

        while epoch <= hm_epochs:
            if epoch != 1:
                this_models_dir = os.path.join(models_dir, str(sorted(list(map(int, os.listdir(models_dir))))[-1]))
                saver.restore(sess, os.path.join(this_models_dir, 'model.ckpt'))
            epoch_loss = 1
            with open(os.path.join(temp_dir, 'lexicon.pickle'), 'rb') as f:
                lexicon = pickle.load(f)
            with open(os.path.join(this_data_dir, 'train_set_shuffled.csv'), buffering=20000, encoding='latin-1') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
                    label = line.split(':::')[0]
                    tweet = line.split(':::')[1]
                    current_words = word_tokenize(tweet.lower())
                    current_words = [lemmatizer.lemmatize(i) for i in current_words]

                    features = np.zeros(len(lexicon))

                    for word in current_words:
                        if word.lower() in lexicon:
                            index_value = lexicon.index(word.lower())
                            features[index_value] += 1
                    line_x = list(features)
                    line_y = eval(label)
                    batch_x.append(line_x)
                    batch_y.append(line_y)
                    if len(batch_x) >= batch_size:
                        _, c = sess.run([optimizer, cost], feed_dict={x: np.array(batch_x),
                                                                      y: np.array(batch_y)})
                        epoch_loss += c
                        batch_x = []
                        batch_y = []
                        batches_run += 1
                        print('Batch run:', batches_run, '/', total_batches, '| Epoch:', epoch, '| Batch Loss:', c)
                    # if batches_run > 50:
                    #     break
            this_model_dir = os.path.join(models_dir, str(epoch))
            if not os.path.exists(this_model_dir):
                os.mkdir(this_model_dir)
            saver.save(sess, os.path.join(this_model_dir, 'model.ckpt'))
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            print('-----------------------------------------------------------------')
            with open(tf_log, 'a') as f:
                f.write(str(epoch) + '\n')
            epoch += 1


def test_neural_network(test_file, models_dir=os.path.join(temp_dir, 'models')):
    prediction = neural_network(x)
    saver = tf.train.Saver(max_to_keep=50)
    for k in sorted(list(map(int, os.listdir(models_dir)))):
        model_file = os.path.join(models_dir, str(k), 'model.ckpt')
        # model_file = os.path.join(
        #    os.path.join(models_dir, str(sorted(list(map(int, os.listdir(models_dir))))[-1])), 'model.ckpt')
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            # try restoring the file
            try:
                saver.restore(sess, model_file)
            except Exception as e:
                print(str(e))

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            with open(test_file, 'rb') as f:
                test_x, test_y = pickle.load(f)
            print('Accuracy for epoch ', k, ': was :', accuracy.eval({x: np.array(test_x), y: np.array(test_y)}))
            break

    #how length is correlated to the sentiment in the movie
    av_pos = 0
    pos_s = 0
    neg_s = 0
    av_neg = 0
    counter = 0
    for this_y in test_y:
        if np.argmax(this_y) == 0:
            #positive
            av_pos += sum(test_x[counter])
            pos_s += 1
        else:
            av_neg += sum(test_x[counter])
            neg_s += 1
        counter += 1
    assert counter == (pos_s + neg_s)
    print('Average length of ', pos_s, ' positive tests cases was ', av_pos/pos_s)
    print('Average length of ', neg_s, ' positive tests cases was ', av_neg/neg_s)



def use_neural_network(input_data, models_dir):
    prediction = neural_network(x)

    saver = tf.train.Saver(max_to_keep=50)
    with open(os.path.join(temp_dir, 'lexicon.pickle'), 'rb') as f:
        lexicon = pickle.load(f)

    model_file = os.path.join(models_dir, 'model.ckpt')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_file)
        current_words = word_tokenize(input_data)
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                features[index_value] += 1

        base_result = prediction.eval(feed_dict={x: [features]})
        result = sess.run(tf.argmax(base_result, 1))

        if result[0] == 0:
            print('Positive:', input_data, base_result)
        elif result[0] == 1:
            print('Negative:', input_data, base_result)


if __name__ == '__main__':
    if to_train:
        train_neural_network(x)

    if to_test:
        model_file = os.path.join(temp_dir, 'model.ckpt')
        # test_file = os.path.join(this_data_dir, 'test_set.csv')
        test_file_preprocessed = os.path.join(this_data_dir, 'test_file_processed.pickle')
        if not os.path.exists(test_file_preprocessed):
            print('Error, test file does not exist')
        else:
            models_dir = os.path.join(temp_dir, 'models')
            test_neural_network(test_file_preprocessed, models_dir)

    if to_usage_testing:
        # using model 11 which showed best test accuracy
        models_dir = os.path.join(temp_dir, 'models', '11')
        use_neural_network('Great', models_dir)
        use_neural_network('I hated this movie, on every level and just having to write this review makes ME want to think about jumping off of a bridge. Avoid this movie if you can, you have been warned. A wonderful movie it is not. ', models_dir)
