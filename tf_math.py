import tensorflow as tf

# Convert the following to TensorFlow:
x = tf.placeholder(tf.int32) #10
y = tf.placeholder(tf.int32) #2
z = tf.subtract(tf.cast(tf.divide(x, y), tf.int32), tf.constant(1))

# Print z from a session
with tf.Session() as sess:
    output = sess.run(z, feed_dict={x: 10, y: 2})

print(output)
