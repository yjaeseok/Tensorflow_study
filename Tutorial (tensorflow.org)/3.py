import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

num_points = 2000
vectors_set = []

for i in range(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.0, 0.9),
                            np.random.normal(0.0, 0.9)])
    else:
        vectors_set.append([np.random.normal(3.0, 0.5),
                            np.random.normal(1.0, 0.5)])

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# df = pd.DataFrame({"x": [v[0] for v in vectors_set],
#                   "y": [v[1] for v in vectors_set]})
# sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
# plt.show()

import tensorflow as tf

vectors = tf.constant(vectors_set)
k = 2
centroid = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k, -1]))

expended_vectors = tf.expand_dims(vectors, 0)
expended_centroid = tf.expand_dims(centroid, 1)

diff = tf.subtract(expended_vectors, expended_centroid)
sqr = tf.square(diff)
distances = tf.reduce_sum(sqr, 2)
assignments = tf.argmin(distances, 0)

means = tf.concat([tf.reduce_mean(tf.gather(vectors,
                                            tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])),
                                  reduction_indices=[1]) for c in range(k)], 0)

update_centroid = tf.assign(centroid, means)

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

for step in range(100):
    _, centroid_values, assignment_values = sess.run([update_centroid, centroid, assignments])

data = {"x": [], "y": [], "cluster": []}

for i in range(len(assignment_values)):
    data["x"].append(vectors_set[i][0])
    data["y"].append(vectors_set[i][1])
    data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
plt.show()
