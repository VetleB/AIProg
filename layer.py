import numpy as np
import tensorflow as tf
import main


class Layer():

    def __init__(self, network, index, invariable, insize, outsize, af, iwr, name=None):
        self.network = network
        self.insize = insize
        self.outsize = outsize
        self.AF = af
        self.iwr = iwr
        self.input = invariable
        self.index = index
        self.name = 'module-'+str(index) if name is None else name
        self.build()

    def build(self):
        mona = self.name
        n = self.outsize
        with tf.name_scope('weights-'+str(self.index)):
            self.weights = tf.Variable(np.random.uniform(self.iwr[0], self.iwr[1], size=(self.insize, n)), name=mona+'-wgt', trainable=True)
            main.summary(self.weights)
        with tf.name_scope('biases-'+str(self.index)):
            self.biases = tf.Variable(np.random.uniform(self.iwr[0], self.iwr[1], size=n), name=mona+'-bias', trainable=True)
            main.summary(self.biases)
        with tf.name_scope('act_func-'+str(self.index)):
            self.pre_out = tf.matmul(self.input, self.weights)+self.biases
            self.output = self.AF(self.pre_out, name=mona+'-out')
            main.summary(self.output)
        self.network.add_module(self)