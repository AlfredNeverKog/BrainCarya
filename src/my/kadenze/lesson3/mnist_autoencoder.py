from mnist import MNIST
import  numpy as np
import tensorflow as tf
from src.my.lib.utils import montage
import matplotlib.pyplot as plt
from PIL import Image

src = '../../../../data/mnist/'
output='./content/1/%s.jpg'


mndata = MNIST(src)
data = np.array(mndata.load_testing())
X = data[0]
Y = data[1]

items = 100
imgs = np.array([i for i in np.array(X[:items])]).reshape(items,28,28)
n_features = 784

n_input = n_features

Y = imgs.reshape(items,n_features).astype(float)
current_input = imgs.reshape(items,n_features).astype(float)

Ws = []
Bs = []
dimensions = [512,256,128,64]


for layer_i,n_ouputs in enumerate(dimensions):
    with tf.variable_scope("encoder/variable/%s" % layer_i):
        W = tf.get_variable(name="weight%s" % layer_i, dtype=tf.float64,
                            initializer=tf.contrib.layers.xavier_initializer(),
                            shape=[n_input, n_ouputs])
        #B = tf.get_variable(name='bias%s' % layer_i, dtype=tf.float64,
        #                    initializer=tf.random_normal_initializer(mean=0.0, stddev=1.1),
        #                    shape=[n_ouputs])
        #h = tf.nn.bias_add(value=tf.matmul(current_input, W),
        #                   bias=B)


        h = tf.matmul(current_input, W)
        current_input = h
        current_input = tf.nn.relu(current_input)

        n_input = n_ouputs
        Ws.append(W)
        #Bs.append()

Ws = Ws[::-1]#reverse
Bs = Bs[::-1]#reverse

#dimensions = dimensions[::1][1:].append(n_features)
dimensions = dimensions[::-1][1:] +[n_features]

#Build DECODER

for layer_i,n_ouputs in enumerate(dimensions):
    with tf.variable_scope("encoder/variable/%s" % layer_i):

                                                            ##128x64 -> 64x128
        h = value=tf.matmul(current_input,tf.transpose(Ws[layer_i]))
        if layer_i + 1 < len(Bs):
            h = tf.nn.bias_add(h,bias=Bs[layer_i + 1])
        current_input = h
        current_input = tf.nn.relu(current_input)

        n_input = n_ouputs


loss_func = tf.reduce_mean(tf.squared_difference(current_input, Y), 1)
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss_func)

counter = 0
with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    for i in range(50000):
        sess.run(train)
        if i % 15 == 0:

            Image.fromarray(montage(sess.run(current_input).reshape(items,28,28)).astype(np.uint8)) \
                .save(output % ("0"*(5 - len(str(counter))) +  str(counter)))
            print(sess.run(tf.reduce_mean(loss_func)))
            counter += 1


