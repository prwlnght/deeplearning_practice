import tensorflow as tf

v1 = tf.Variable(tf.zeros([2, 2], dtype=tf.float32, name='v1'))
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(v1))
    save_path = saver.save(sess, './model.ckpt')
    print("model saved in file:", save_path)

    # Create an op to increment v1, run it, and print the result.
    increment_op = v1.assign_add(tf.ones([2, 2]))
    sess.run(increment_op)
    print(sess.run(v1))

    # Restore from the checkpoint saved above.
    saver.restore(sess, './model.ckpt')
    print(sess.run(v1))


