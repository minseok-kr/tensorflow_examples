# 선형회귀 기본 문제
#
# X가 5일때 Y가 52
# X가 7일때 Y가 72 라면
#
# X가 8일때 Y의 값은?
#
import tensorflow as tf

# 학습 데이터
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# 변수 선언
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 가설
hypothesis = X * W + b

# 최적화
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# 실행
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(5000):
    v_cost, v_W, v_b, _ = sess.run([cost, W, b, train], feed_dict={X: [5, 7], Y: [52, 72]})
    if step % 500 == 0:
        print(step, v_cost, v_W, v_b)

# 예측
print("예측 Y: ", sess.run(hypothesis, feed_dict={X: [8]}))
