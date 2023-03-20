from ops import *


class Weights(object):
    def __init__(self, scope=None):
        self.weights = {}
        self.scope = scope
        self.B = 16
        self.F = 64
        self.global_step = tf.placeholder(tf.int32, shape=[], name="global_step")
        self.bias_initializer = tf.constant_initializer(value=0.0)
        self.scale = 4
        self.scaling_factor = 0.1
        self.PS = 3 * (4 * 4)  # channels x scale^2
        self.xavier = tf.contrib.layers.xavier_initializer()
        self.build_CNN_params()
        print('Initialize weights {}'.format(self.scope))

    def build_CNN_params(self):
        #tf.variable_scope
        with tf.compat.v1.variable_scope(self.scope):
            self.resFilters = list()
            self.resBiases = list()

            for i in range(0, (self.B * 2)):
                self.weights['conv%d/w' % (i)] = tf.get_variable('conv%d/kernel' % (i), shape=[3, 3, self.F, self.F],
                                                                 initializer=self.xavier)
                self.weights['conv%d/b' % (i)] = tf.get_variable('conv%d/bias' % (i), shape=[64],
                                                                 initializer=self.bias_initializer)
                self.resFilters.append(self.weights['conv%d/w' % (i)])
                self.resBiases.append(self.weights['conv%d/b' % (i)])

            self.weights['conv64/w'] = tf.get_variable('conv64/kernel', shape=[3, 3, 3, self.F],
                                                       initializer=self.xavier)
            self.weights['conv64/b'] = tf.get_variable('conv64/bias', shape=[self.F], initializer=self.bias_initializer)

            self.weights['conv65/w'] = tf.get_variable('conv65/kernel', shape=[3, 3, self.F, self.F],
                                                       initializer=self.xavier)
            self.weights['conv65/b'] = tf.get_variable('conv65/bias', shape=[self.F], initializer=self.bias_initializer)

            self.weights['conv66/w'] = tf.get_variable('conv66/kernel', shape=[3, 3, self.F, self.PS],
                                                       initializer=self.xavier)
            self.weights['conv66/b'] = tf.get_variable('conv66/bias', shape=[self.PS],
                                                       initializer=self.bias_initializer)


class MODEL(object):
    def __init__(self, name):
        self.name = name
        self.scale = 4
        self.B = 16
        print('Build Model {}'.format(self.name))

    def resBlock(self, inpt, f_nr, param):
        x = conv2d(inpt, param['conv%d/w' % (f_nr)], param['conv%d/b' % (f_nr)], scope='conv%d/b' % (f_nr),
                   activation='ReLU')

        x = conv2d(x, param['conv%d/w' % (f_nr + 1)], param['conv%d/b' % (f_nr + 1)], scope='conv%d/b' % (f_nr + 1))
        # x = x * self.scaling_factor
        x = x * 0.1

        return inpt + x

    def forward(self, x, param):
        self.input=x

        self.param = param

        with tf.variable_scope(self.name):
            # first conv
            x = conv2d(self.input, param['conv64/w'], param['conv64/b'], scope='conv64')
            out1 = tf.identity(x)

            # all residual blocks
            for i in range(self.B):
                x = self.resBlock(x, (i * 2), self.param)

            # last conv
            x = conv2d(x, param['conv65/w'], param['conv65/b'], scope='conv65')
            x = x + out1

            # upsample via sub-pixel, equivalent to depth to space
            x = conv2d(x, param['conv66/w'], param['conv66/b'], scope='conv66')
            self.output = tf.nn.depth_to_space(x, self.scale, data_format='NHWC', name="NHWC_output")











