import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
import tflowtools as TFT
import mnist.mnist_basics as mb
import numpy.random as NPR

from mnist import mnist_basics

def main(dimensions, HAF, OAF, loss_function, learn_rate, IWR, optimizer, data_source, case_fraction,
         validation_fraction, validation_interval, test_fraction, minibatch_size, map_batch_size, steps, layers,
         map_dendrograms, display_weights, display_biases):
    pass


class Network():
    # Set-up
    def __init__(self, dims, learn_rate=0.01, mbs=10, vfrac=0.1, tfrac=0.1, haf=tf.nn.relu, oaf=tf.nn.softmax):
        self.caseman = Caseman('parity', test_fraction=tfrac, validation_fraction=vfrac)
        self.learn_rate = learn_rate
        self.dims = dims
        self.grabvars = []
        self.global_training_step = 0
        self.minibatch_size = mbs if mbs else dims[0]
        self.HAF = haf
        self.OAF = oaf
        self.modules = []
        self.build()

    def add_module(self, module):
        self.modules.append(module)

    # Create network
    def build(self):
        tf.reset_default_graph()
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
        layer = Layer(self, len(self.dims), invar, insize, self.dims[-1], af=self.OAF)
        invar = layer.output
        insize = layer.outsize

        self.output = layer.output
        # if self.OAF:
        #     self.output = tf.nn.softmax(self.output)
        self.target = tf.placeholder(tf.float64, shape=(None, layer.outsize), name='Target')
        self.configure_learning()

    def configure_learning(self):
        self.error = tf.reduce_mean(tf.square(self.target-self.output), name='MSE')
        self.predictor = self.output
        optimizer = tf.train.GradientDescentOptimizer(self.learn_rate)
        self.trainer = optimizer.minimize(self.error, name='Backprop')

    # Do training
    def do_training(self, sess, cases, epochs=100, continued=False):
        if not(continued): self.error_history = []
        for i in range(epochs):
            error = 0
            step = self.global_training_step + i
            gvars = [self.error]
            mbs = self.minibatch_size
            ncases = len(cases)
            nmb=math.ceil(ncases/mbs)
            for cstart in range(0, ncases, mbs):
                cend = min(ncases, cstart+mbs)
                minibatch = cases[cstart:cend]
                inputs = [c[0] for c in minibatch]
                targets = [c[1] for c in minibatch]
                feeder = {self.input: inputs, self.target: targets}
                _,grabvals,_ = self.run_one_step([self.trainer], gvars, session=sess, feed_dict=feeder, step=step)
                error += grabvals[0]
            self.error_history.append((step, error/nmb))
        self.global_training_step += epochs

    def training_session(self, epochs, sess=None, dir='probeview', continued=False):
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.current_session = session
        self.do_training(session, self.caseman.get_training_cases(), epochs, continued)

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

    def run(self, epochs=100, sess=None, continued=False, bestk=None):
        self.training_session(epochs, sess=sess, continued=continued)
        self.test_on_trains(sess=self.current_session, bestk=bestk)
        self.testing_session(sess=self.current_session, bestk=bestk)
        #self.close_current_session(view=False)
    # Graph stuff

class Layer():

    def __init__(self, network, index, invariable, insize, outsize, af):
        self.network = network
        self.insize = insize
        self.outsize = outsize
        self.AF = af
        self.input = invariable
        self.index = index
        self.name = "Module-"+str(index)
        self.build()

    def build(self):
        mona = self.name
        n = self.outsize
        self.weights = tf.Variable(np.random.uniform(-.1, .1, size=(self.insize, n)), name=mona+'-wgt', trainable=True)
        self.biases = tf.Variable(np.random.uniform(-.1, .1, size=n), name=mona+'-bias', trainable=True)
        self.output = self.AF(tf.matmul(self.input, self.weights)+self.biases, name=mona+'-out')
        self.network.add_module(self)


class Caseman():
    def __init__(self, case, test_fraction=0.1, validation_fraction=0.1):
        self.case_generator = create_case_generator(case)
        self.cases = self.case_generator()

        self.test_fraction = test_fraction
        self.validation_fraction=validation_fraction
        self.train_fraction = 1-(validation_fraction+test_fraction)

        self.organize_cases()


    def organize_cases(self):
        ca = np.array(self.cases)
        NPR.shuffle(ca)
        sep1 = round(len(self.cases)*self.test_fraction)
        sep2 = sep1 + round(len(self.cases)*self.validation_fraction)
        self.test_cases = ca[0:sep1]
        self.validation_cases = ca[sep1:sep2]
        self.train_cases = ca[sep2:]

    def get_training_cases(self): return self.train_cases

    def get_validation_cases(self): return self.validation_cases

    def get_testing_cases(self): return self.test_cases


def create_case_generator(case):
    if case == 'parity':
        case_gen = (lambda : TFT.gen_all_parity_cases(num_bits=10))
    elif case == 'symmetry':
        case_gen = (lambda : [[c[:-1], [c[-1]]] for c in TFT.gen_symvect_dataset(vlen=101, count=2000)])
    elif case == 'auto':
        case_gen = (lambda : TFT.gen_all_one_hot_cases(len=32))
    elif case == 'bit_count':
        case_gen = (lambda : TFT.gen_vector_count_cases(num=500, size=15))
    elif case == 'segment_count':
        case_gen = (lambda : TFT.gen_segmented_vector_cases(vectorlen=25, count=1000, minsegs=0, maxsegs=8))
    elif case == 'mnist':
        all_cases = [[c[:-1], [c[-1]]] for c in mb.load_all_flat_cases(unify=True)]
        NPR.shuffle(all_cases)
        cut_off = int(len(all_cases)/10)
        fraction_of_cases = all_cases[:cut_off]
        case_gen = (lambda : fraction_of_cases)
    elif case in ['wine', 'yeast', 'glass']:
        case_gen = (lambda : get_all_irvine_cases(case=case))
    elif case == 'hc':
        #TODO: hacker's choice
        #case_gen = (lambda kwargs: TFT.gen_vector_count_cases(**kwargs))
        case_gen = create_case_generator('parity')
    else:
        raise ValueError('No such case')
    return case_gen

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

def autoexec(nbits=4, epochs=300, lrate=0.03, mbs=None, vfrac=0.1, tfrac=0.1, bestk=None, sm=False):
    net = Network([2**nbits, nbits, 2**nbits], learn_rate=lrate, vfrac=vfrac, tfrac=tfrac, mbs=mbs, softmax=sm)
    net.run(epochs, bestk=bestk)
