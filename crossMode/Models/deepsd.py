from Models.ops import *

class Weights(object):
    def __init__(self, scope=None):
        self.weights={}
        self.scope=scope
        self.kernel_initializer=tf.variance_scaling_initializer()

        self.build_CNN_params()
        print('Initialize weights {}'.format(self.scope))


    def build_CNN_params(self):
        kernel_initializer=self.kernel_initializer
        with tf.variable_scope(self.scope):
            self.weights['conv1/w'] = tf.get_variable('conv1/kernel', [9, 9, 3, 64], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv1/b'] = tf.get_variable('conv1/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv2/w'] = tf.get_variable('conv2/kernel', [1, 1, 64, 32], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv2/b'] = tf.get_variable('conv2/bias',[32], dtype=tf.float32, initializer=tf.zeros_initializer())


            self.weights['conv3/w'] = tf.get_variable('conv3/kernel', [5, 5, 32, 3], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv3/b'] = tf.get_variable('conv3/bias',[3], dtype=tf.float32, initializer=tf.zeros_initializer())



class MODEL(object):
    def __init__(self, name):
        self.name = name
        print('Build Model {}'.format(self.name))

    def forward(self, x, param):
        self.input=x
        self.param=param
        with tf.variable_scope(self.name):
            self.conv1 = conv2d(self.input, param['conv1/w'], param['conv1/b'], scope='conv1', activation='ReLU')
            self.conv2 = conv2d(self.conv1, param['conv2/w'], param['conv2/b'], scope='conv2', activation='ReLU')
            self.conv3 = conv2d(self.conv2, param['conv3/w'], param['conv3/b'], scope='conv3')
            self.output = self.conv3

