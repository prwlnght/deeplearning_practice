import tensorflow as tf
import os
import skimage
from skimage import transform
from skimage.color import rgb2gray
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
import random


def show_stats(images, labels):
    # show histogram of label distribution
    plt.hist(labels, 62)
    plt.show()
    plt.cla()

    # show random images to glean into the data_set
    traffic_signs = [300, 2250, 3650, 4000]

    unique_labels = set(labels)
    images28 = [transform.resize(image, (28, 28)) for image in images]
    images28 = np.array(images28)
    images28 = rgb2gray(images28)

    for i in range(len(traffic_signs)):
        # plt.cla()
        plt.subplot(1, 4, i + 1)
        plt.axis('off')
        plt.imshow(images28[traffic_signs[i]], cmap="gray")
        plt.subplots_adjust(wspace=0.5)
        print("shape:{0}, min:{1}, max: {2}".format(images28[traffic_signs[i]].shape, images28[traffic_signs[i]].min(),
                                                    images28[traffic_signs[i]].max()))
    plt.show()
    plt.cla()

    plt.figure(figsize=(15, 15))
    counter = 1

    for label in unique_labels:
        # pick the first image for each label
        image = images28[labels.index(label)]

        # define 64 subpolots
        plt.subplot(8, 8, counter)
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        counter += 1
        plt.imshow(image)

    # plt.show()


def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []

    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels


ROOT_PATH = "C:\\Users\\ppaudyal\\workspace\\tfflow_practice\\data"
train_data_directory = os.path.join(ROOT_PATH, 'TrafficSigns\\Training')
test_data_directory = os.path.join(ROOT_PATH, 'TrafficSigns\\Testing')

images, labels = load_data(train_data_directory)

unique_lables = set(labels)
images28 = images
images28 = [transform.resize(image, (28, 28)) for image in images]
images28 = np.array(images28)
images28 = rgb2gray(images28)

x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
y = tf.placeholder(dtype=tf.int32, shape=[None])

# flatten the images
images_flat = tf.contrib.layers.flatten(x)

# fully connected layer
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

# define an optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print('images_flat', images_flat)
print('logits: ', logits)
print('loss:', loss)
print('predicted_labels: ', correct_pred)

tf.set_random_seed(1234)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(201):
    print('EPOCH', i)
    _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
    if i % 10 == 0:
        print('Accuracy: ', accuracy_val)
    print('Done with Epoch')

# Evaluation
# sample_indexes = random.sample(range(len(images28)), 100)
# sample_images = [images28[i] for i in sample_indexes]
# sample_labels = [labels[i] for i in sample_indexes]
#
# predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
#
# # Print th ereal and predicted labels
# print(sample_labels)
#print(predicted)

# # Display the predictions and the ground truth visually.
# fig = plt.figure(figsize=(10, 10))
# counter=1
# for i in range(len(sample_images)):
#     truth = sample_labels[i]
#     prediction = predicted[i]
#     if truth == prediction:
#         if counter >= 20:
#             break
#         plt.subplot(5, 4, counter)
#         plt.axis('off')
#         color = 'green' if truth == prediction else 'red'
#         plt.title("T: {0}P:{1}".format(truth, prediction),
#                  color=color)
#         plt.imshow(sample_images[i], cmap='gray')
#         counter += 1
#
# print ('total: {0}, this: {1}'.format(i, counter))
# plt.show()

test_images, test_labels = load_data(test_data_directory)

#transform the images to 28 by 28
test_images28 = [transform.resize(image, (28, 28)) for image in test_images]
test_images28 = np.array(test_images28)
test_images28 = rgb2gray(test_images28)

#run predictions
predicted = sess.run([correct_pred], feed_dict={x:test_images28})[0]

match_count = sum([int(y==y_)for y, y_ in zip(test_labels, predicted)])

accuracy = match_count/len(test_labels)

#print the accuracy
print('Accuracy: {:.3f}'.format(accuracy))

sess.close()