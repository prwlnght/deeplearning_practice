import tensorflow as tf


#initializae two variables

x1 = tf.constant([1,2,4,5])
x2 = tf.constant([5,6,7,9])

#x3 = tf.placeholder([tf.float64])

result = tf.multiply(x1,x2)

sess = tf.Session()
config = tf.ConfigProto(log_device_placement=True)

with tf.Session() as sess:
    output = sess.run(result)
    print(output)
