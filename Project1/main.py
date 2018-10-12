import tensorflow as tf
import matplotlib.pyplot as PLT
import os
from Project1 import network, caseman as cman, tflowtools as TFT


# Set up a case manager, create a network, call run and plot training and validation history
def main(dims=[], data_source=(), steps=0, optimizer='gd', loss_func='mse', l_rate=0.1, HAF=tf.nn.relu, OAF=tf.nn.relu,
         IWR=(-.1, .1), case_fraction=1, vint=1000, vfrac=0.1, tfrac=0.1, minibatch_size=64, map_batch_size=0,
         map_layers=[], map_dendrograms=[], display_weights=[], display_biases=[], premade=None):

    # Delete old files
    os.system('del /Q /F .\probeview')

    caseman = cman.Caseman(casefunc=data_source[0], kwargs=data_source[1], case_fraction=case_fraction,
                           test_fraction=tfrac, validation_fraction=vfrac)

    net = network.Network(dims, caseman, steps, l_rate, minibatch_size, HAF, OAF, loss_func, optimizer, vint,
                          map_batch_size, IWR=IWR, map_layers=map_layers, map_dendrograms=map_dendrograms,
                          display_weights=display_weights, display_biases=display_biases)

    net.run(bestk=net.bestk)
    TFT.plot_training_history(error_hist=net.error_history, validation_hist=net.validation_history)

    PLT.show()
    return net


# Read datasets from UC Irvine
# txt files must be in same directory as file
def get_irvine_cases(case='wine', **kwargs):
    file_dict = {'wine': ('wine.txt', 6),
                 'yeast': ('yeast.txt', 10),
                 'glass': ('glass.txt', 7),
                 'iris': ('iris.txt', 3)}
    f = open(file_dict[case][0])
    feature_target_vector = []
    for line in f.readlines():
        line = line.strip('\n')
        nums = line.split(',')
        features = [float(x) for x in nums[:-1]]
        if case=='wine':
            clazz = one_hotify(float(nums[-1])-3, file_dict[case][1])
        else:
            clazz = one_hotify(float(nums[-1])-1, file_dict[case][1])
        feature_target_vector.append([features, clazz])
    f.close()
    if case in ['glass', 'iris', 'wine']:
        scale(feature_target_vector)
    # print(feature_target_vector)
    return feature_target_vector

# Turn int into a one_hot vector
def one_hotify(clazz, num_clazzes):
    one_hot = [0]*num_clazzes
    one_hot[int(clazz)] = 1
    return one_hot


# Scale datasets
# Used when features have large differences in magnitudes
def scale(dataset):
    fmax = max([max(d[0]) for d in dataset])
    fmin = min([min(d[0]) for d in dataset])

    for pair in dataset:
        features = pair[0]
        for i in range(len(features)):
            f_old = features[i]
            f_new = (f_old-fmin)/(fmax-fmin)
            features[i] = f_new



# Not used at the moment
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
