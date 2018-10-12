import tensorflow as tf
import numpy as np
from Project1 import tflowtools as TFT, layer
import numpy.random as NPR


class Network():
    # Set-up
    def __init__(self, dims, caseman, steps, learn_rate, mbs, haf, oaf, loss, optimizer, vint, mpb_size,
                 IWR, map_layers, map_dendrograms, display_weights, display_biases):
        self.caseman = caseman
        self.learn_rate = learn_rate
        self.dims = dims
        self.steps = steps

        self.grabvars = []
        self.global_training_step = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.minibatch_size = mbs if mbs else dims[0]

        self.HAF = haf
        self.OAF = oaf
        self.iwr = IWR
        self.loss_func = loss
        self.opt = optimizer
        self.bestk = 1
        self.modules = []

        self.error = 0
        self.error_history = []
        self.accuracy_history = []
        self.validation_interval = vint
        self.validation_history = []

        self.map_batch_size = mpb_size
        self.map_layers = map_layers
        self.map_dendrograms = map_dendrograms
        self.display_weights = display_weights
        self.display_biases = display_biases

        self.current_session = None
        self.log_dir = "probeview"
        self.build()


    # Add layer to list of layers
    def add_module(self, module):
        self.modules.append(module)

    # Create network
    def build(self):
        tf.reset_default_graph()

        self.current_session = tf.Session()

        # Build input layer
        num_inputs = self.dims[0]
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name='input')
        invar = self.HAF(self.input)
        insize=num_inputs

        # Build hidden layers
        for i, outsize in enumerate(self.dims[1:-1], 1):
            hidden_layer = layer.Layer(self, i, invar, insize, outsize, af=self.HAF, iwr=self.iwr)
            invar = hidden_layer.output
            insize = hidden_layer.outsize

        # Build output layer
        output_layer = layer.Layer(self, 'output', invar, insize, self.dims[-1], af=self.OAF, name='output', iwr=self.iwr)

        self.output = output_layer.output
        self.preout = output_layer.pre_out
        self.target = tf.placeholder(tf.float64, shape=(None, output_layer.outsize), name='Target')

        self.configure_learning()

    def configure_learning(self):
        # Define loss function
        if self.loss_func=='mse':
            self.error = tf.losses.mean_squared_error(labels=self.target, predictions=self.preout)
        elif self.loss_func=='x_entropy':
            self.error = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target, logits=self.preout, name='x_entropy')
            self.error = tf.reduce_mean(self.error)

        self.predictor = self.output

        # Define optimizer
        if self.opt == 'gd':
            optimizer = tf.train.GradientDescentOptimizer(self.learn_rate)
        elif self.opt == 'rms':
            optimizer = tf.train.RMSPropOptimizer(self.learn_rate)
        elif self.opt == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learn_rate)
        elif self.opt == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(self.learn_rate)
        self.trainer = optimizer.minimize(self.error, name='Backprop')

    # Do training
    def do_training(self, sess, cases, continued=False):
        if not(continued): self.error_history = []

        steps_left = self.steps
        num_mb = len(self.caseman.get_training_cases()) // self.minibatch_size
        step = 0

        # Loop through all the steps
        while steps_left > 0:
            # steps left, cut short during final iteration if needed
            num_mb = num_mb if steps_left > self.minibatch_size else steps_left

            # Total error over minibatch
            error = 0

            gvars = [self.error]

            # Run some minibatches
            for j in range(num_mb):
                NPR.shuffle(cases)
                minibatch = cases[0:self.minibatch_size]

                inputs = [c[0] for c in minibatch]
                targets = [c[1] for c in minibatch]
                feeder = {self.input: inputs, self.target: targets}

                result,grabvals = self.current_session.run([self.trainer, gvars], feed_dict=feeder)

                error += grabvals[0]

                # Validation at intervals
                if ((step+j)%self.validation_interval == 0):
                    self.validation(step)

            step += num_mb
            avg_error = error/num_mb
            self.error_history.append((step, avg_error))

            steps_left -= num_mb

        # Final validation run
        self.validation(step)

        self.global_training_step.assign_add(step)

    def validation(self, step):
        if not self.caseman.validation_fraction == 0:
            correct = self.do_testing(sess=self.current_session, cases=self.caseman.get_validation_cases(), bestk=self.bestk, msg=None)
            acc = correct/len(self.caseman.get_validation_cases())
            self.validation_history.append((step, 1-acc))

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
            # This basically always happens
            self.test_func = self.gen_match_counter(self.predictor, targets, k=bestk)

        testres, grabvals = self.current_session.run([self.test_func, self.grabvars], feed_dict=feeder)

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
        labels = [list(l) for l in labels]

        # Find indices of top values and see if they're the same
        correct = tf.equal(tf.nn.top_k(logits, k)[1], tf.nn.top_k(labels, 1)[1])

        # Cast booleans to ints and sum
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def run(self, sess=None, continued=False, bestk=1):
        tf.global_variables_initializer()
        session = sess if sess else TFT.gen_initialized_session(dir=self.log_dir)
        self.current_session = session
        self.training_session(sess=self.current_session, continued=continued)
        self.test_on_trains(sess=self.current_session, bestk=bestk, msg='Total training')
        if not self.caseman.test_fraction == 0:
            self.testing_session(sess=self.current_session, bestk=bestk)
        if not self.map_batch_size == 0:
            self.visualize(sess=self.current_session)

    # Perform mapping, dendrograms, etc. if applicable
    def visualize(self, sess):
        cases = self.caseman.get_mapping_cases(self.map_batch_size)

        map_vars = self.get_map_vars()
        weights = self.get_weights()
        biases = self.get_biases()

        data = self.mapping_session(sess, cases, map_vars, weights, biases)

        if not (self.map_dendrograms is None or self.map_dendrograms == []):
            self.create_dendrogram(cases, data)

        if not (self.display_weights is None or self.display_weights == []):
            self.visualize_weights(data)

        if not (self.display_biases is None or self.display_biases == []):
            self.visualize_biases(data)

        TFT.hinton_plot(np.array([c[1] for c in cases]), title='Target')

    def visualize_weights(self, data):
        for w in self.display_weights:
            matrix = data[2][w-1]
            TFT.display_matrix(matrix, title='Weights-'+str(w))

    def visualize_biases(self, data):
        for b in self.display_biases:
            matrix = data[3][b-1]
            TFT.display_matrix(np.array([matrix]), title='Biases-'+str(b))

    def create_dendrogram(self, cases, data):

        for layer in self.map_dendrograms:
            labels = [TFT.bits_to_str(c[1]) for c in cases]
            dendro = data[1][layer]

            dendro_label_pairs = [[dendro[i], labels[i]] for i in range(len(dendro))]

            dendro, labels = self.remove_dupes(dendro_label_pairs)

            TFT.dendrogram(features=dendro, labels=labels, title='Dendrogram-'+str(layer))

    def remove_dupes(self, list_with_dupes):
        labels = []
        no_dupes = []

        for e in list_with_dupes:
            if e[1] not in labels:
                labels.append(e[1])
                no_dupes.append(e[0])

        return no_dupes, labels

    def get_weights(self):
        weights = []
        for w in self.display_weights:
            weights.append(self.modules[w-1].weights)
        return weights

    def get_biases(self):
        biases = []
        for b in self.display_biases:
            biases.append(self.modules[b-1].biases)
        return biases

    def get_map_vars(self):
        sorted(self.map_layers)
        if 0 in self.map_layers:
            map_vars = [self.input]
            layers = self.map_layers[1:]
        else:
            map_vars = []
            layers = self.map_layers[:]

        for layer_num in layers:
            if layer_num >= 1:
                map_vars.append(self.modules[layer_num-1].output)

        return map_vars

    def mapping_session(self, sess, cases, map_vars, weights, biases):
        inputs = [c[0] for c in cases]
        feeder = {self.input: inputs}
        data = self.current_session.run([self.predictor, map_vars, weights, biases], feed_dict=feeder)
        map_vars = data[1]
        for i in range(len(self.map_layers)):
            TFT.hinton_plot(map_vars[i], title='Layer '+str(self.map_layers[i]))
        return data

    def test(self, input):
        out = self.current_session.run([self.predictor], feed_dict={self.input:input})
        return out
