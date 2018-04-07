import tensorflow as tf

# None은 크기가 정해지지 않았음을 의미합니다.
X = tf.placeholder(tf.float32, [None, 3])
print(X)

x_data = [[1, 2, 3], [4, 5, 6]]

W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([2, 1]))

expr = tf.matmul(X, W) + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("=== x_data ===")
print(x_data)
print("=== W ===")
print(sess.run(W))
print("=== b ===")
print(sess.run(b))
print("=== expr ===")
print(sess.run(expr, feed_dict={X: x_data}))

sess.close()
