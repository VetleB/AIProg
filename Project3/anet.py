from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import optimizers
import random
import numpy

class Anet:

    def __init__(self, layers, model_name, haf='tanh', oaf='tanh', loss='mean_squared_error', optimizer='sgd', lrate=0.01, pre_train_epochs=250, load_existing=False):

        opts = {'sgd': optimizers.SGD
            ,'adagrad': optimizers.Adagrad
            ,'adam': optimizers.Adam
            ,'rms': optimizers.RMSprop}

        self.path = 'saved_anets/'
        self.file_name = model_name
        self.pre_train_epochs = pre_train_epochs
        self.model = None

        if load_existing:
            self.load_model()
        if not self.model:
            self.model = Sequential()
            self.model.add(Dense(layers[1], activation=haf, input_dim=layers[0]))
            for layer in layers[2:-1]:
                self.model.add(Dense(layer, activation=haf))
            self.model.add(Dense(layers[-1], activation=oaf))
            optmzr = opts[optimizer](lrate)
            self.model.compile(loss=loss, optimizer=optmzr, metrics=['accuracy'])

    def train_on_cases(self, cases, epochs=100):
        features, targets = self.feature_target(cases)

        self.model.fit(features, targets, epochs=epochs, batch_size=5, verbose=0)

    def accuracy(self, cases):
        features, targets = self.feature_target(cases)

        scores = self.model.evaluate(features, targets)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

    def feature_target(self, vector):
        features = numpy.array([f_t[0] for f_t in vector])
        targets = numpy.array([f_t[1] for f_t in vector])

        return features, targets

    def distribution(self, state):
        state = numpy.array(state)
        prediction = self.model.predict(state)

        return prediction

    def normalize(self, vector):
        vector_sum = sum(vector)
        if vector_sum == 0:
            return list(vector)
        return [float(i)/vector_sum for i in vector]

    def load_model(self):
        try:
            loaded_model = load_model(self.path + self.file_name + '.h5')
            self.model = loaded_model
        except Exception as e:
            print(e)
            print("Couldn't load model, here's a new one")

    def save_model(self, name=None):
        file_name = name if name else self.file_name
        self.model.save(self.path + file_name + '.h5')

    def topp_save(self, batch):
        topp_name = self.file_name + '_topp_' + str(batch)
        self.model.save(self.path + topp_name + '.h5')
        return topp_name
