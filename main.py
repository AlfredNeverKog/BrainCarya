import numpy as np
import tensorflow as tf
from neural_network import NeuralNetwork


with tf.Session() as sess:

    n = NeuralNetwork(input=NeuralNetwork.generate_layer(5,tf.sigmoid),output=NeuralNetwork.generate_layer(1,tf.sigmoid),
                      hiden_layers=[NeuralNetwork.generate_layer(i,tf.sigmoid) for i in range(5,7)],
                      session=sess)


    def get_mean(data):
        return [np.mean(i) for i in data]


    X = np.random.rand(100, 5)
    Y = get_mean(X)
    Y = np.array(Y).reshape(100, 1)

    n.train(X,Y,epochs=3000)

    # testing
    test = [[.1, .3,.1,.2,.3]]
    test_ans = get_mean(test)

    layers = n.create_layers(tf.constant(test, dtype=tf.float32))
    print("test ans", test_ans)
    print("predicated", sess.run(layers[-1]))