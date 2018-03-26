import numpy as np
import tensorflow as tf

# [털, 날개]
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

y_data = np.array([
    [1, 0, 0],  # 기타
    [0, 1, 0],  # 포유류
    [0, 0, 1],  # 조류
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
W2 = tf.Variable(tf.random_uniform([10, 3], -1., 1.))

b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))

L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))

model = tf.add(tf.matmul(L1, W2), b2)

# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# 텐서플로의 세션을 초기화합니다.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 앞서 구성한 특징과 레이블 데이터를 이용해 학습을 100번 진행합니다.
for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    # 학습 도중 10번에 한 번씩 손실값을 출력해봅니다.
    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print("예측값:", sess.run(prediction, feed_dict={X: x_data}))
print("실제값:", sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("정확도: %.2f" % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
