import numpy as np
from mnist import MNIST
from sklearn.preprocessing import normalize
import tensorflow as tf


import src.my.lib.filter as filter
from src.my.lib.image import Image,ArrayImage
import src.my.lib.convolution as conv
import src.my.lib.pooling as pooling
import src.my.lib.convolution_network as conv_net


#Load MNIST
mndata = MNIST('../../data/mnist')
input = mndata.load_testing()
images = [np.array(a).reshape(28,28) for a in input[0]]
labels = input[1]


#Trim
images = images[:310]
labels = labels[:310]

print ("Create Convulution")
#Create convulution
convolutions = [conv_net.ConvolutionNetwork(ArrayImage(i)).forward()
                for i in images]
#reshaper
convolutions = np.array([res.reshape(1,np.array(res.shape).prod()) for res in convolutions])

#normalize
convolutions = np.array([x / np.linalg.norm(x) for x in convolutions])

#reshapein (outr shape = (10,1,4)) -> (10,4)
convolutions = convolutions.reshape([convolutions.shape[0],convolutions.shape[2]])

print("Training")
from src.my.lib.neural_network import NeuralNetwork

with tf.Session() as sess:
    n = NeuralNetwork(input=NeuralNetwork.generate_layer(convolutions[0].shape[0],tf.nn.relu)
                      ,output=NeuralNetwork.generate_layer(10,tf.nn.relu),
                      hiden_layers=[NeuralNetwork.generate_layer(i,tf.sigmoid) for i in range(7,14)],
                      session=sess)



    Y = np.array(labels)


    #Convert digits to vector when 1 represent answer (2 -> [0,0,1,0,0,0,0,0,0,0]
    Y = np.array([[1 if i == q else 0 for i in range(10)] for q in Y])

    n.train(convolutions,Y,epochs=300000)

    #test
    print("TESING")
    layers = n.create_layers(tf.constant(convolutions, dtype=tf.float32))
    predicted = sess.run(layers[-1])

    print(np.array(np.array([np.argmax(i) for i in predicted]) == np.array(labels)).sum()," of ",len(labels))




#images = np.array(a[0][0]).reshape(28,28)
"""
ArrayImage(images[0]).print()

conv_netw = conv_net.ConvolutionNetwork(Image("../../data/x.png"))
res = conv_netw.forward()
res = res.reshape(1,np.array(res.shape).prod())

"""


"""
for y in range(image.height()):
    for x in range(image.width()):
        char = '+' if image.pixel(y,x) == 1 else ' '
        print(char, end="", flush=True)
    print()

"""
