import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import misc
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
from src.my.lib.utils import montage_filters,montage
model, labels = ('./inception5h/tensorflow_inception_graph.pb',
                 './inception5h/imagenet_comp_graph_label_strings.txt')
import time


with gfile.GFile(model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

tf.import_graph_def(graph_def=graph_def,name='inception')
#[print(a.name) for a in g.get_operations()]

g = tf.get_default_graph()
im = np.array(misc.imresize(Image.open('./data/coffee.jpg'),(200,200))[:,:,:3])
im_4d = im[np.newaxis]
x = g.get_tensor_by_name('inception/input:0')
def Classification():
    last_layer = g.get_tensor_by_name('inception/output2:0')


    with tf.Session() as sess:
        res = np.squeeze(sess.run(last_layer,feed_dict={x: im_4d}))
        txt = open(labels).readlines()
        labels_val = [(key, val.strip()) for key, val in enumerate(txt)]
        print([(labels_val[i], res[i]) for i in res.argsort()[::-1][:5]])

def ConvFilters():
    with tf.Session() as sess:
        W = g.get_tensor_by_name('inception/conv2d0_w:0')
        W_eval = sess.run(W)
        Ws = np.array([montage_filters(W_eval[:, :, [i], :]) for i in range(3)])
        a = np.rollaxis(Ws, 0, 3) #(n1,n2,n3) -> (n3,n2,n1)
        #Filters
        plt.imshow(((a / np.max(np.abs(a))) * 128 + 128).astype(np.uint), interpolation='nearest')
        plt.show()

def Convs():
    with tf.Session() as sess:
        tensor = g.get_tensor_by_name('inception/conv2d0_pre_relu:0')
        start = time.time()

        conv =sess.run(tensor,feed_dict={x:im_4d})
        print('conv_time %s'%(time.time() - start))

        print(conv.shape)
        start = time.time()
        mtage = montage(np.array([conv[0,:,:,i] for i in range(64)] ))
        print('montage_time %s' % (time.time() - start))
        plt.imshow(mtage)
        plt.show()


Convs()