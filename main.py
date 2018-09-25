import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
import tflowtools as TFT
import mnist.mnist_basics as mb
import numpy.random as NPR
import os

from mnist import mnist_basics

def main(dimensions, HAF, OAF, loss_function, learn_rate, IWR, optimizer, data_source, case_fraction,
         validation_fraction, validation_interval, test_fraction, minibatch_size, map_batch_size, steps, layers,
         map_dendrograms, display_weights, display_biases):
    casefunc, kwargs = data_source
    caseman = Caseman(casefunc, kwargs, case_fraction, test_fraction, validation_fraction)
    net = Network(dimensions, caseman, steps, learn_rate, minibatch_size, HAF, OAF, loss_function, optimizer,
                  validation_interval, map_batch_size)
    net.run()


class Network():
    # Set-up
    def __init__(self, dims, caseman, steps, learn_rate=0.01, mbs=10, haf=tf.nn.relu, oaf=tf.nn.softmax,
                 loss=tf.reduce_mean, optimizer=tf.train.GradientDescentOptimizer, vint=None, mb_size=0):
        self.caseman = caseman
        self.learn_rate = learn_rate
        self.dims = dims
        self.grabvars = []
        self.steps = steps
        self.global_training_step = 0
        self.minibatch_size = mbs if mbs else dims[0]
        self.HAF = haf
        self.OAF = oaf
        self.loss_func = loss
        self.opt = optimizer(learning_rate=learn_rate)
        self.validation_interval = vint
        self.map_batch_size = mb_size
        self.modules = []
        self.current_session = None
        self.log_dir = ""
        self.build()

    def add_module(self, module):
        self.modules.append(module)

    # Create network
    def build(self):
        tf.reset_default_graph()

        self.current_session = TFT.gen_initialized_session(dir=dir)
        self.writer = tf.summary.FileWriter(self.log_dir, graph=self.current_session.graph)

        num_inputs = self.dims[0]
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name='input')
        invar = self.input
        insize=num_inputs

        # Build hidden layers
        for i, outsize in enumerate(self.dims[1:-1], 1):
            layer = Layer(self, i, invar, insize, outsize, af=self.HAF)
            invar = layer.output
            insize = layer.outsize

        # Build output layer
        layer = Layer(self, len(self.dims)-1, invar, insize, self.dims[-1], af=self.OAF, name='output')
        invar = layer.output
        insize = layer.outsize

        self.output = layer.output
        # if self.OAF:
        #     self.output = tf.nn.softmax(self.output)
        self.target = tf.placeholder(tf.float64, shape=(None, layer.outsize), name='Target')
        self.configure_learning()

    def configure_learning(self):
        # Define loss function
        if self.loss_func==tf.reduce_mean:
            self.error = self.loss_func(tf.square(self.target - self.output), name='MSE')
        else:
            self.error = self.loss_func(tf.square(self.target - self.output), name='MSE')


        tf.scalar_summary("error", self.error)

        self.predictor = self.output
        optimizer = tf.train.GradientDescentOptimizer(self.learn_rate)
        self.trainer = optimizer.minimize(self.error, name='Backprop')

    # Do training
    def do_training(self, sess, cases, continued=False):
        if not(continued): self.error_history = []
        trainables = tf.trainable_variables()

        acc_vars = [tf.Variable(trainable.initialized_value(), trainable=False) for trainable in trainables]

        zeros = [trainable.assign(tf.zeros_like(trainable)) for trainable in acc_vars]

        gradients = self.opt.compute_gradients(loss=self.error)

        acc_ops = [acc_vars[i].assign_add(gradient[0]) for i, gradient in enumerate(gradients)]

        apply = self.opt.apply_gradients([(acc_vars[i], trainable) for i, trainable in enumerate(tf.trainable_variables())])

        steps_left = self.steps
        num_mb = len(self.caseman.get_training_cases()) // self.minibatch_size
        step = 0

        while steps_left > 0:
            num_mb = num_mb if steps_left > self.minibatch_size else steps_left
            error = 0

            gvars = [self.error]

            sess.run(zeros)

            for j in range(num_mb):
                NPR.shuffle(cases)
                minibatch = cases[0:self.minibatch_size]

                inputs = [c[0] for c in minibatch]
                targets = [c[1] for c in minibatch]
                feeder = {self.input: inputs, self.target: targets}

                _,grabvals,_ = self.run_one_step([acc_ops], gvars, session=sess, feed_dict=feeder, step=step)
                error += grabvals[0]
                if ((step+j)%self.validation_interval==0):
                    print('error:', error)

            sess.run([apply])
            # self.run_one_step([apply])

            step += num_mb
            avg_error = error/self.minibatch_size
            self.error_history.append((step, avg_error))

            steps_left -= num_mb
        self.global_training_step += step

    def training_session(self, sess=None, dir='probeview', continued=False):
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.do_training(session, self.caseman.get_training_cases(), continued)

    # Do testing
    def do_testing(self, sess, cases, msg='Testing', bestk=None):
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target:targets}
        self.test_func = self.error
        if bestk is not None:
            self.test_func = self.gen_match_counter(self.predictor, [TFT.one_hot_to_int(list(v)) for v in targets], k=bestk)
        testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, session=sess, feed_dict=feeder)
        if bestk is None:
            print('%s Set error = %f' % (msg, testres))
        else:
            print('%s Set correct classification = %f %%' % (msg, 100*(testres/len(cases))))
        return testres

    def testing_session(self, sess, bestk=None):
        cases = self.caseman.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(sess, cases, msg='Final testing', bestk=bestk)

    def test_on_trains(self, sess, bestk=None):
        self.do_testing(sess, self.caseman.get_training_cases(), msg='Total training', bestk=bestk)

    def gen_match_counter(self, logits, labels, k=1):
        correct = tf.nn.in_top_k(tf.cast(logits, tf.float32), labels, k)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    # Run
    def run_one_step(self, operators, grabbed_vars=None, dir='probeview', session=None, feed_dict=None, step=1, show_interval=1):
        sess = session if session else TFT.gen_initialized_session(dir=dir)

        results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)

        return results[0], results[1], sess

    def run(self, sess=None, continued=False, bestk=None, dir='probeview'):
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.current_session = session
        self.training_session(sess=self.current_session, continued=continued)
        self.test_on_trains(sess=self.current_session, bestk=bestk)
        self.testing_session(sess=self.current_session, bestk=bestk)
        #self.close_current_session(view=False)
    # Graph stuff

class Layer():

    def __init__(self, network, index, invariable, insize, outsize, af, name=None):
        self.network = network
        self.insize = insize
        self.outsize = outsize
        self.AF = af
        self.input = invariable
        self.index = index
        self.name = 'module-'+str(index) if name is None else name
        self.build()

    def build(self):
        mona = self.name
        n = self.outsize
        self.weights = tf.Variable(np.random.uniform(-.1, .1, size=(self.insize, n)), name=mona+'-wgt', trainable=True)
        self.biases = tf.Variable(np.random.uniform(-.1, .1, size=n), name=mona+'-bias', trainable=True)
        self.output = self.AF(tf.matmul(self.input, self.weights)+self.biases, name=mona+'-out')
        self.network.add_module(self)


class Caseman():
    def __init__(self, casefunc, kwargs, case_fraction=1, test_fraction=0.1, validation_fraction=0.1):
        self.cases = casefunc(**kwargs)
        self.case_fraction = case_fraction

        self.test_fraction = test_fraction
        self.validation_fraction=validation_fraction
        self.train_fraction = 1-(validation_fraction+test_fraction)

        self.organize_cases()


    def organize_cases(self):
        ca = np.array(self.cases)
        # NPR.shuffle(ca)
        ca = ca[:round(self.case_fraction*len(ca))]
        self.cases = ca
        sep1 = round(len(ca)*self.test_fraction)
        sep2 = sep1 + round(len(ca)*self.validation_fraction)
        self.test_cases = ca[0:sep1]
        self.validation_cases = ca[sep1:sep2]
        self.train_cases = ca[sep2:]

    def get_training_cases(self): return self.train_cases

    def get_validation_cases(self): return self.validation_cases

    def get_testing_cases(self): return self.test_cases


# def create_case_generator(case):
#     if case == 'parity':
#         case_gen = (lambda : TFT.gen_all_parity_cases(num_bits=10))
#     elif case == 'symmetry':
#         case_gen = (lambda : [[c[:-1], [c[-1]]] for c in TFT.gen_symvect_dataset(vlen=101, count=2000)])
#     elif case == 'auto':
#         case_gen = (lambda : TFT.gen_all_one_hot_cases(len=32))
#     elif case == 'bit_count':
#         case_gen = (lambda : TFT.gen_vector_count_cases(num=500, size=15))
#     elif case == 'segment_count':
#         case_gen = (lambda : TFT.gen_segmented_vector_cases(vectorlen=25, count=1000, minsegs=0, maxsegs=8))
#     elif case == 'mnist':
#         all_cases = [[c[:-1], [c[-1]]] for c in mb.load_all_flat_cases(unify=True)]
#         NPR.shuffle(all_cases)
#         cut_off = int(len(all_cases)/10)
#         fraction_of_cases = all_cases[:cut_off]
#         case_gen = (lambda : fraction_of_cases)
#     elif case in ['wine', 'yeast', 'glass']:
#         case_gen = (lambda : get_all_irvine_cases(case=case))
#     elif case == 'hc':
#         #TODO: hacker's choice
#         #case_gen = (lambda kwargs: TFT.gen_vector_count_cases(**kwargs))
#         case_gen = create_case_generator('parity')
#     else:
#         raise ValueError('No such case')
#     return case_gen

def get_all_irvine_cases(case='wine', **kwargs):
    file_dict = {'wine': 'wine.txt',
                 'yeast': 'yeast.txt',
                 'glass': 'glass.txt'}
    f = open(file_dict[case])
    feature_target_vector = []
    for line in f.readlines():
        line = line.strip('\n')
        nums = line.split(';') if case=='wine' else line.split(',')
        features = [float(x) for x in nums[:-1]]
        wine_class = [float(nums[-1])]
        feature_target_vector.append([features, wine_class])
    f.close()
    return feature_target_vector


def autoexec(steps=5000, lrate=0.03, mbs=32, casefunc=TFT.gen_vector_count_cases, kwargs={'num':1000, 'size':15}, vfrac=0.1, tfrac=0.1, bestk=None, sm=False):
    os.system('del /Q /F .\probeview')
    caseman = Caseman(casefunc, kwargs, test_fraction=tfrac, validation_fraction=vfrac)
    net = Network([15, 4, 16], caseman, steps, learn_rate=lrate, mbs=mbs, vint=1000)
    net.run(bestk=bestk)
    os.system('start chrome http://desktop-1vusl9o:6006')
    os.system('tensorboard --logdir=probeview')

