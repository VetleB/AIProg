import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
import tflowtools as TFT
import mnist.mnist_basics as mb
import numpy.random as NPR
import os

import network
import parameters
import caseman as cman

from mnist import mnist_basics

def main(dims=[], data_source=(), steps=0, optimizer='gd', loss_func='mse', l_rate=0.1, HAF=tf.nn.relu, OAF=tf.nn.relu,
         IWR=(-.1, .1), bestk=1, case_fraction=1, vint=1000, vfrac=0.1, tfrac=0.1, minibatch_size=64, map_batch_size=0,
         map_layers=[], map_dendrograms=[], display_weights=[], display_biases=[], sm=False, eint=1, premade=None):

    os.system('del /Q /F .\probeview')

    if premade is not None:
        params = parameters.good_parameters(premade)
        return main(**params)
    else:
        caseman = cman.Caseman(casefunc=data_source[0], kwargs=data_source[1], case_fraction=case_fraction,
                               test_fraction=tfrac, validation_fraction=vfrac)

        net = network.Network(dims, caseman, steps, l_rate, minibatch_size, HAF, OAF, sm, loss_func, optimizer, vint,
                              eint, map_batch_size, bestk, IWR=IWR, map_layers=map_layers,
                              map_dendrograms=map_dendrograms, display_weights=display_weights,
                              display_biases=display_biases)

        net.run(bestk=bestk)
        TFT.plot_training_history(error_hist=net.error_history, validation_hist=net.validation_history)
        # TODO: create dendrograms
        # TFT.dendrogram(features=, labels=)
        # TFT.plot_training_history(net.accuracy_history, ytitle='% correct', title='Accuracy')
        PLT.show()
        return net




# Graph stuff
def summary(variable):
    with tf.name_scope('summary'):
        mean = tf.reduce_mean(variable)
        tf.summary.scalar('mean', mean)








def get_all_irvine_cases(case='wine', **kwargs):
    file_dict = {'wine': ('wine.txt', 8),
                 'yeast': ('yeast.txt', 10),
                 'glass': ('glass.txt', 7)}
    f = open(file_dict[case][0])
    feature_target_vector = []
    for line in f.readlines():
        line = line.strip('\n')
        nums = line.split(';') if case=='wine' else line.split(',')
        features = [float(x) for x in nums[:-1]]
        clazz = one_hotify(float(nums[-1])-1, file_dict[case][1])
        feature_target_vector.append([features, clazz])
    f.close()
    return feature_target_vector

def one_hotify(clazz, num_clazzes):
    one_hot = [0]*num_clazzes
    one_hot[int(clazz)] = 1
    return one_hot


def autoexec(dims=[], steps=50000, lrate=0.05, mbs=64, loss='mse', opt='gd', vint=1000, eint=100, casefunc=TFT.gen_vector_count_cases, kwargs={'num':500, 'size':15}, vfrac=0.1, tfrac=0.1, bestk=None, sm=False):
    os.system('del /Q /F .\probeview')
    caseman = cman.Caseman(casefunc, kwargs, test_fraction=tfrac, validation_fraction=vfrac)
    net = network.Network(dims, caseman, steps, learn_rate=lrate, mbs=mbs, vint=vint, eint=eint, loss=loss, bestk=bestk, softmax=sm, optimizer=opt)
    net.run(bestk=bestk)
    TFT.plot_training_history(error_hist=net.error_history, validation_hist=net.validation_history)
    # TODO: create dendrograms
    #TFT.dendrogram(features=, labels=)
    #TFT.plot_training_history(net.accuracy_history, ytitle='% correct', title='Accuracy')
    PLT.show()
    return net
    #Desktop
    #os.system('start chrome http://desktop-1vusl9o:6006
    #Laptop
    #os.system('start chrome http://DESKTOP-D5MC4MC:6006')
    #os.system('tensorboard --logdir=probeview')
