from ops import *

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
            self.weights['conv1/w'] = tf.get_variable('conv1/kernel', [3, 3, 3, 64], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv1/b'] = tf.get_variable('conv1/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv2/w'] = tf.get_variable('conv2/kernel', [3, 3, 64, 64], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv2/b'] = tf.get_variable('conv2/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())


            self.weights['conv3/w'] = tf.get_variable('conv3/kernel', [3, 3, 64, 64], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv3/b'] = tf.get_variable('conv3/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv4/w'] = tf.get_variable('conv4/kernel', [3, 3, 64, 64], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv4/b'] = tf.get_variable('conv4/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv5/w'] = tf.get_variable('conv5/kernel', [3, 3, 64, 64], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv5/b'] = tf.get_variable('conv5/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv6/w'] = tf.get_variable('conv6/kernel', [3, 3, 64, 64], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv6/b'] = tf.get_variable('conv6/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv7/w'] = tf.get_variable('conv7/kernel', [3, 3, 64, 64], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv7/b'] = tf.get_variable('conv7/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv8/w'] = tf.get_variable('conv8/kernel', [3, 3, 64, 3], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv8/b'] = tf.get_variable('conv8/bias',[3], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv9/w'] = tf.get_variable('conv9/kernel', [3, 3, 3, 27], initializer=kernel_initializer,dtype=tf.float32)
            self.weights['conv9/b'] = tf.get_variable('conv9/bias', [27], dtype=tf.float32,initializer=tf.zeros_initializer())

            self.weights['conv10/w'] = tf.get_variable('conv10/kernel', [3, 3, 3, 27], initializer=kernel_initializer,dtype=tf.float32)
            self.weights['conv10/b'] = tf.get_variable('conv10/bias', [27], dtype=tf.float32,initializer=tf.zeros_initializer())

            self.weights['conv11/w'] = tf.get_variable('conv11/kernel', [3, 3, 27, 27], initializer=kernel_initializer,dtype=tf.float32)
            self.weights['conv11/b'] = tf.get_variable('conv11/bias', [27], dtype=tf.float32,initializer=tf.zeros_initializer())


class MODEL(object):
    def __init__(self, name):
        self.name = name
        self.scale = 3
        print('Build Model {}'.format(self.name))

    def forward(self, x, param):
        self.input=x
        self.param=param
        with tf.variable_scope(self.name):
            self.conv1 = conv2d(self.input, param['conv1/w'], param['conv1/b'], scope='conv1', activation='ReLU')
            self.head = self.conv1
            for idx in range(2,8):
                self.head = conv2d(self.head, param['conv%d/w' %idx], param['conv%d/b' % idx], scope='conv%d' %idx, activation='ReLU')

            self.out1 = conv2d(self.head, param['conv8/w'], param['conv8/b'], scope='conv8', activation=None)
            self.output = tf.add(self.input, self.out1)

            x = conv2d(self.output, param['conv9/w'], param['conv9/b'], scope='conv9')
            self.deconv_1 = tf.nn.depth_to_space(x, self.scale, data_format='NHWC', name="NHWC_output")
            self.conv_1 = conv2d(self.deconv_1, param['conv10/w'], param['conv10/b'],strides=3, scope='conv10')
            # y = tf.subtract(x, self.conv_1)
            y = tf.add(x, self.conv_1)

            self.z = conv2d(y, param['conv11/w'], param['conv11/b'], scope='conv11')
            self.deconv_2 = tf.nn.depth_to_space(self.z, self.scale, data_format='NHWC', name="NHWC_output")

            self.output = tf.add(self.deconv_1, self.deconv_2)
