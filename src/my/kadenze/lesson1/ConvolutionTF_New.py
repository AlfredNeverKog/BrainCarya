import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image


sess = tf.InteractiveSession()

#image = Image.open('./data/she_reshaped.png')
#im_with_one_chanel = np.array(image)[:, :, 0]
#Initalize placeholders
img = tf.placeholder(tf.float32, shape=[None, None], name="image")
img_3d = tf.expand_dims(img, 2) #append 1 dims at end(2 - index)
img_4d = tf.expand_dims(img_3d, 0)

mean = tf.placeholder(tf.float32, name="mean")
sigma = tf.placeholder(tf.float32, name="sigma")
ksize = tf.placeholder(tf.int32, name='kernel_size')

x = tf.linspace(-3.0,3.0,ksize)
y =  tf.contrib.distributions.Normal(mu=mean, sigma=sigma).pdf(x)
y_2d = tf.matmul(tf.reshape(y,[ksize,1]), tf.reshape(y,[1,ksize]))
kernel = tf.reshape(y_2d,shape=[ksize,ksize,1,1])

conv = tf.nn.conv2d(img_4d,filter=kernel,strides=[1,1,1,1],padding="SAME")

plt.imshow(conv.eval(feed_dict={
    img: np.array(Image.open('./data/she_reshaped.png'))[:,:,0],
    mean: 0.0,
    sigma: 1.0,
    ksize: 50
}).squeeze(), cmap='gray')

plt.show()


print(img_4d)

