# NN is a network of neurons > neurons work


import tensorflow as tf

x1 = tf.constant([[3, 2], [1, 5], [5, 1]])
x2 = tf.constant([[3, 0, 0], [1, 0, 0]])

# to actually get the results run a session

with tf.Session() as sess:
    with tf.device("/cpu:0"):
        # print(x1.shape)
        # print(x2)
        v = tf.get_variable("v", shape= (2,1), initializer=tf.random_normal_initializer())
        assignment = v.assign_add([[1],[2]])
        tf.global_variables_initializer().run()
        print(sess.run(assignment))  # or assignment.op.run()
