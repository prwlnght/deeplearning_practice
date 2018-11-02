import tensorflow as tf
import os
import skimage
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt


def show_stats(images, labels):
    # print a his
    # for i in range(0, 2):
    #     plt.subplot(1, 2, i+1)
    #     plt.axis('off')
    #     plt.imshow(images[i])
    #     plt.subplots_adjust(wspace=0.1)
    # plt.show()
    # return

    # print one of each image

    print(labels)
    unique_lables = set(labels)
    counter = 1;
    images28 = images
    images28 = [transform.resize(image, (280, 280)) for image in images]
    images28 = np.array(images28)
    images28 = rgb2gray(images28)

    for label in unique_lables:
        print('unique labels {0}'.format(label))
        # make a 2*2 window
        plt.subplot(2, 2, counter)
        plt.imshow(images28[labels.index(label)], cmap='gray')
        plt.axis('off')
        plt.title(
            "Label {0} ({1}), size: {2}".format(label, labels.count(label), str(images28[labels.index(label)].shape)))
        counter += 1
    plt.show()


def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []

    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".png")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels


ROOT_PATH = "C:\\Users\\ppaudyal\\workspace\\tfflow_practice\\data"
train_data_directory = os.path.join(ROOT_PATH, 'AslSigns\\Training')
test_data_directory = os.path.join(ROOT_PATH, 'AslSigns\\Testing')

images, labels = load_data(train_data_directory)
show_stats(images, labels)

unique_lables = set(labels)
images28 = images
images28 = [transform.resize(image, (280, 280)) for image in images]
images28 = np.array(images28)
images28 = rgb2gray(images28)

x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
y = tf.placeholder(dtype=tf.int32, shape=[None])

# flatten the images
images_flat = tf.contrib.layers.flatten(x)

# fully connected layer
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

# define an optimizer
train_op = tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(loss)

#convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

#define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print('images_flat', images_flat)
print('logits: '. logits)
print('loss:', loss)
print('predicted_labels: ', correct_pred)


