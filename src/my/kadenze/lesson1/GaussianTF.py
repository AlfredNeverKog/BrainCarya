import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
sess = tf.InteractiveSession() #open eval


sigma = 3.0
mean = 0

def gaus(x):
    y = (1 / (sigma * tf.sqrt(2.0 * 3.14))) * tf.exp(tf.neg((tf.pow(x - mean, 2.0)) / (2 * tf.pow(sigma, 2.0))))
    return y
def geus2d():
    x = tf.linspace(-5.0,5.0,3)
    y = gaus(x)
    plt.plot(x.eval(), y.eval())
    plt.show()

def gaus3d():
    x = tf.linspace(-5.0, 5.0, 150)
    y = gaus(x)
    size = x.get_shape().as_list()[0]

    gaus2d = tf.matmul(tf.reshape(y, [size, 1]), tf.reshape(y, [1, size]))
    plt.imshow(gaus2d.eval())
    plt.show()
def animation():
    from matplotlib import animation
    import random

    fig = plt.figure()
    ax = plt.axes()
    line = ax.imshow([[]])
    def animate(size):
        global mean
        print
        size, mean
        size = 300
        mean += ((random.random() / 5) * (-1.0 if random.random() > .5 else 1.0))

        x = tf.linspace(-5.0, 5.0, size + 1)

        y = (1 / (sigma * tf.sqrt(2.0 * 3.14))) * tf.exp(tf.neg((tf.pow(x - mean, 2.0)) / (2 * tf.pow(sigma, 2.0))))
        size = x.get_shape().as_list()[0]

        gaus2d = tf.matmul(tf.reshape(y, [size, 1]), tf.reshape(y, [1, size]))
        val = gaus2d.eval()
        return ax.imshow(val),

    """
    animate quality
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=60, interval=1, blit=True)
    anim.save('gausian_quality.mp4', fps=3, extra_args=['-vcodec', 'libx264'])
    plt.show()

    sigma = 1.0
    mean = 0.0
    """

    """
    animate(5)
    anim = animation.FuncAnimation(fig, animate,
                                   frames=20, interval=1, blit=True)
    anim.save('gausian_move.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
    plt.show()
    """
gaus3d()