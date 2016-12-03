import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image


"""
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print (sess.run(c))
"""
def reshape_image_and_save(location, img_format):
    image = Image.open("%s.%s"%(location, img_format))
    im_array = np.array(image)
    mMax = min(im_array.shape[:2])

    reshaped = im_array[:mMax,:mMax,:]

    Image.fromarray(reshaped).save("%s_reshaped.%s"%(location, img_format))


    print(mMax)
#reshape_image_and_save('./data/she','png')

sess = tf.InteractiveSession()

image = Image.open('./data/she_reshaped.png')
#plt.imshow(np.array(image)[:,:,0], cmap='gray')
#plt.show()

im_with_one_chanel = np.array(image)[:, :, 0]
im_arr = tf.constant(im_with_one_chanel.squeeze(), dtype=tf.float32)

#Kernel
x_1d = tf.linspace(-3., 3., 30)
z_1d = tf.contrib.distributions.Normal(mu=0.0, sigma=1.0).pdf(x_1d)
z_size = x_1d.get_shape().as_list()[0]
z_2d = tf.matmul(tf.reshape(z_1d,[z_size,1]),tf.reshape(z_1d,[1,z_size]))


plt.figure(1)
plt.imshow(z_2d.eval())
plt.figure(2)
plt.plot(x_1d.eval(),z_1d.eval())

tf.initialize_all_variables()

z_size = x_1d.get_shape().as_list()[0]

#Convert to 4d dimensiopn
z_4d = tf.reshape(z_2d, [z_size, z_size, 1, 1])
image4d = tf.reshape(im_arr,[1,im_arr.get_shape().as_list()[0],
                             im_arr.get_shape().as_list()[0],1])

convolved = tf.nn.conv2d(image4d, z_4d, strides=[1,1,1,1], padding='SAME')
print(z_4d.eval())

conv = convolved.eval().squeeze()
#normalize
conv = conv/float(conv.max()) * 255.0

print(conv)
plt.figure(3)

plt.imshow(conv,cmap='gray')

plt.show()

#plt.imshow(image,cmap='gray')

