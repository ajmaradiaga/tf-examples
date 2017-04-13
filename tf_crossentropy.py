import tensorflow as tf

softmax_data = [0.7, 0.2, 0.1] #yhat
one_hot_data = [1.0, 0.0, 0.0] #y

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# Print cross entropy from session
# sum of y*ln(yhat)
ce = -tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))

output = None

with tf.Session() as sess:
    output = sess.run(ce, feed_dict={one_hot: one_hot_data, softmax:softmax_data})

print("Cross Entropy of y: {} and yhat: {} is {}".format(one_hot_data, softmax_data, output))
