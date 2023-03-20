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
            self.weights['conv1/w'] = tf.get_variable('conv1/kernel', [3, 3, 3, 128], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv1/b'] = tf.get_variable('conv1/bias',[128], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv2/w'] = tf.get_variable('conv2/kernel', [3, 3, 128, 128], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv2/b'] = tf.get_variable('conv2/bias',[128], dtype=tf.float32, initializer=tf.zeros_initializer())


            self.weights['conv3/w'] = tf.get_variable('conv3/kernel', [3, 3, 128, 27], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv3/b'] = tf.get_variable('conv3/bias',[27], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv4/w'] = tf.get_variable('conv4/kernel', [3, 3, 3, 128], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv4/b'] = tf.get_variable('conv4/bias',[128], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv5/w'] = tf.get_variable('conv5/kernel', [3, 3, 128, 27], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv5/b'] = tf.get_variable('conv5/bias',[27], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv6/w'] = tf.get_variable('conv6/kernel', [3, 3, 3, 128], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv6/b'] = tf.get_variable('conv6/bias',[128], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv7/w'] = tf.get_variable('conv7/kernel', [3, 3, 128, 27], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv7/b'] = tf.get_variable('conv7/bias',[27], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv8/w'] = tf.get_variable('conv8/kernel', [3, 3, 3, 128], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv8/b'] = tf.get_variable('conv8/bias',[128], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv9/w'] = tf.get_variable('conv9/kernel', [3, 3, 128, 27], initializer=kernel_initializer,dtype=tf.float32)
            self.weights['conv9/b'] = tf.get_variable('conv9/bias', [27], dtype=tf.float32,initializer=tf.zeros_initializer())

            self.weights['conv10/w'] = tf.get_variable('conv10/kernel', [3, 3, 3, 3], initializer=kernel_initializer,dtype=tf.float32)
            self.weights['conv10/b'] = tf.get_variable('conv10/bias', [3], dtype=tf.float32,initializer=tf.zeros_initializer())



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
            self.conv2 = conv2d(self.conv1, param['conv2/w'], param['conv2/b'], scope='conv2', activation='ReLU')

            self.conv3 = conv2d(self.conv2, param['conv3/w'], param['conv3/b'], scope='conv3',activation='ReLU')
            self.deconv_1 = tf.nn.depth_to_space(self.conv3, self.scale, data_format='NHWC', name="NHWC_output")
            self.conv4 = conv2d(self.deconv_1, param['conv4/w'], param['conv4/b'], scope='conv4',strides=3,activation='ReLU')
            self.conv4 = tf.subtract(self.conv4, self.conv2)

            self.conv5 = conv2d(self.conv4, param['conv5/w'], param['conv5/b'], scope='conv5',activation='ReLU')
            self.deconv_2 = tf.nn.depth_to_space(self.conv5, self.scale, data_format='NHWC', name="NHWC_output")
            self.conv5 = tf.add(self.deconv_1, self.deconv_2)

            self.conv6 = conv2d(self.conv5, param['conv6/w'], param['conv6/b'], scope='conv6',strides=3,activation='ReLU')
            self.conv7 = conv2d(self.conv6, param['conv7/w'], param['conv7/b'], scope='conv7',activation='ReLU')
            self.deconv_3 = tf.nn.depth_to_space(self.conv7, self.scale, data_format='NHWC', name="NHWC_output")
            self.conv7 = tf.subtract(self.conv5, self.deconv_3)

            self.conv8 = conv2d(self.conv7, param['conv8/w'], param['conv8/b'], scope='conv8',strides=3,activation='ReLU')
            self.conv8 = tf.add(self.conv8, self.conv6)
            self.conv9 = conv2d(self.conv8, param['conv9/w'], param['conv9/b'], scope='conv9',activation='ReLU')
            self.deconv_4 = tf.nn.depth_to_space(self.conv9, self.scale, data_format='NHWC', name="NHWC_output")

            self.output1 = tf.add(self.deconv_4, self.deconv_3)

            self.output = conv2d(self.output1, param['conv10/w'], param['conv10/b'], scope='conv10')


