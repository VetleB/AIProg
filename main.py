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
    def __init__(self, dims, caseman, steps, learn_rate=0.01, mbs=10, haf=tf.nn.relu, oaf=tf.nn.relu,
                 softmax=True, loss='mse', optimizer=tf.train.GradientDescentOptimizer,
                 vint=None, eint=1, mb_size=0, bestk=1):
        self.caseman = caseman
        self.learn_rate = learn_rate
        self.dims = dims
        self.grabvars = []
        self.steps = steps
        self.global_training_step = 0
        self.minibatch_size = mbs if mbs else dims[0]
        self.HAF = haf
        self.OAF = oaf
        self.softmax = softmax
        self.loss_func = loss
        self.opt = optimizer(learning_rate=learn_rate)
        self.bestk=bestk
        self.error = 0
        self.error_interval = eint
        self.error_history = []
        self.accuracy_history = []
        self.validation_interval = vint
        self.validation_history = []
        self.map_batch_size = mb_size
        self.modules = []
        self.current_session = None
        self.log_dir = "probeview"
        self.build()


    def add_module(self, module):
        self.modules.append(module)

    # Create network
    def build(self):
        tf.reset_default_graph()

        #self.current_session = TFT.gen_initialized_session(dir=self.log_dir)
        self.current_session = tf.Session()
        self.writer = tf.summary.FileWriter(self.log_dir, graph=self.current_session.graph)

        num_inputs = self.dims[0]
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name='input')
        invar = self.input
        insize=num_inputs

        # Build hidden layers
        for i, outsize in enumerate(self.dims[1:-1], 1):
            with tf.name_scope('hidden-'+str(i)):
                layer = Layer(self, i, invar, insize, outsize, af=self.HAF)
                invar = layer.output
                insize = layer.outsize

        # Build output layer
        with tf.name_scope('output'):
            layer = Layer(self, 'output', invar, insize, self.dims[-1], af=self.OAF, name='output')

        self.output = layer.output
        if self.softmax and not self.loss_func=='x_entropy': self.output = tf.nn.softmax(self.output)

        # if self.OAF:
        #     self.output = tf.nn.softmax(self.output)
        self.target = tf.placeholder(tf.float64, shape=(None, layer.outsize), name='Target')
        self.configure_learning()

        self.merge_all = tf.summary.merge_all()

    def configure_learning(self):
        # Define loss function
        with tf.name_scope('error'):
            if self.loss_func=='mse':
                self.error = tf.reduce_mean(tf.squared_difference(self.target, self.output), name='MSE')
                self.post_run_error_handling = lambda x : x
            elif self.loss_func=='x_entropy':
                self.error = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target, logits=self.output, name='MSE')
                self.post_run_error_handling = lambda l : sum(l)
            summary(self.error)

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
        show_step = 0

        self.test_trains_and_log(step)

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

                summary,result,grabvals = self.current_session.run([self.merge_all, self.trainer, gvars], feed_dict=feeder)
                self.writer.add_summary(summary, step+j)
                #_,grabvals,sess = self.run_one_step([self.merge_all, acc_ops], gvars, session=sess, feed_dict=feeder, step=step)
                #print(grabvals[0])
                error += self.post_run_error_handling(grabvals[0])
                #print(j, error)
                if ((step+j)%self.validation_interval==0):
                    self.validation(step)
                    #print('error:', error)
                show_step += 1

            #sess.run([apply])
            # self.run_one_step([apply])

            step += num_mb
            avg_error = error/num_mb
            if show_step > self.error_interval:
                self.test_trains_and_log(step)
                show_step = 0
                # correct = self.test_on_trains(sess=self.current_session, bestk=self.bestk)
                # acc = correct/len(self.caseman.get_training_cases())
                # self.accuracy_history.append((step, acc))
            #self.error_history.append((step, avg_error))

            steps_left -= num_mb
        self.validation(step)
        print(self.error_history[-1])
        self.global_training_step += step

    def validation(self, step):
        correct = self.do_testing(sess=self.current_session, cases=self.caseman.get_validation_cases(), bestk=self.bestk, msg='Validation')
        acc = correct/len(self.caseman.get_validation_cases())
        self.validation_history.append((step, 1-acc))

    def test_trains_and_log(self, step):
        correct = self.test_on_trains(sess=self.current_session, bestk=self.bestk, msg=None)
        acc = correct/len(self.caseman.get_training_cases())
        self.error_history.append((step, 1-acc))


    def training_session(self, sess=None, continued=False):
        session = sess if sess else TFT.gen_initialized_session(dir=self.log_dir)
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
        #print(testres)
        if bestk is None and msg is not None:
            print('%s Set error = %f' % (msg, testres))
        elif msg is not None:
            acc = 100*(testres/len(cases))
            print('%s Set correct classification = %f %%' % (msg, acc))
        return testres

    def testing_session(self, sess, bestk=None):
        cases = self.caseman.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(sess, cases, msg='Final testing', bestk=bestk)

    def test_on_trains(self, sess, bestk=None, msg='Training'):
        return self.do_testing(sess, self.caseman.get_training_cases(), msg=msg, bestk=bestk)

    def gen_match_counter(self, logits, labels, k=1):
        correct = tf.nn.in_top_k(tf.cast(logits, tf.float32), labels, k)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    # Run
    def run_one_step(self, operators, grabbed_vars=None, session=None, feed_dict=None, step=1, show_interval=1):
        sess = session if session else TFT.gen_initialized_session(dir=self.log_dir)

        results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)

        return results[0], results[1], sess

    def run(self, sess=None, continued=False, bestk=None):
        tf.global_variables_initializer()
        session = sess if sess else TFT.gen_initialized_session(dir=self.log_dir)
        #print(self.caseman.cases)
        self.current_session = session
        self.training_session(sess=self.current_session, continued=continued)
        self.test_on_trains(sess=self.current_session, bestk=bestk, msg='Total training')
        self.testing_session(sess=self.current_session, bestk=bestk)
        #self.close_current_session(view=False)

# Graph stuff
def summary(variable):
    with tf.name_scope('summary'):
        mean = tf.reduce_mean(variable)
        tf.summary.scalar('mean', mean)



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
        with tf.name_scope('weights-'+str(self.index)):
            self.weights = tf.Variable(np.random.uniform(-.1, .1, size=(self.insize, n)), name=mona+'-wgt', trainable=True)
            summary(self.weights)
        with tf.name_scope('biases-'+str(self.index)):
            self.biases = tf.Variable(np.random.uniform(-.1, .1, size=n), name=mona+'-bias', trainable=True)
            summary(self.biases)
        with tf.name_scope('act_func-'+str(self.index)):
            self.output = self.AF(tf.matmul(self.input, self.weights)+self.biases, name=mona+'-out')
            summary(self.output)
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


def autoexec(steps=50000, lrate=0.05, mbs=64, loss='mse', vint=1000, eint=100, casefunc=TFT.gen_vector_count_cases, kwargs={'num':500, 'size':15}, vfrac=0.1, tfrac=0.1, bestk=None, sm=False):
    os.system('del /Q /F .\probeview')
    caseman = Caseman(casefunc, kwargs, test_fraction=tfrac, validation_fraction=vfrac)
    net = Network([25, 20, 9], caseman, steps, learn_rate=lrate, mbs=mbs, vint=vint, eint=eint, loss=loss, bestk=bestk)
    net.run(bestk=bestk)
    TFT.plot_training_history(error_hist=net.error_history, validation_hist=net.validation_history)
    TFT.dendrogram(features=, labels=)
    #TFT.plot_training_history(net.accuracy_history, ytitle='% correct', title='Accuracy')
    PLT.show()
    #Desktop
    #os.system('start chrome http://desktop-1vusl9o:6006
    #Laptop
    #os.system('start chrome http://DESKTOP-D5MC4MC:6006')
    #os.system('tensorboard --logdir=probeview')
