# Solution is available in the other "solution.py" tab
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = tf.placeholder(tf.int32) #10
y = tf.placeholder(tf.int32) #2
z = tf.subtract(tf.cast(tf.divide(x, y), tf.int32), tf.constant(1))

# TODO: Print z from a session
with tf.Session() as sess:
    output = sess.run(z, feed_dict={x: 10, y: 2})

print(output)
