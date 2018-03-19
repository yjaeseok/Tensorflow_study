import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

# load data
num_points = 1000
vectors_set = []

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.90)
    y1 = x1 * 0.1 * 0.3 + np.random.normal(0.0, 0.01)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

# graph
import matplotlib.pyplot as plt

# plt.plot(x_data, y_data, 'ro')
# plt.legend()
# plt.show()

# tensor

import tensorflow as tf

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(10):
    sess.run(train)

print(sess.run(W), sess.run(b))

plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
