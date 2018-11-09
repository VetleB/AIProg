from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy

class Anet:

    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(120, activation='tanh', input_dim=34))
        self.model.add(Dense(64, activation='tanh'))
        self.model.add(Dense(16, activation='tanh'))
        self.model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['accuracy'])

    def train_on_cases(self, cases, epochs=1):
        features = numpy.array([case[0] for case in cases])
        targets = numpy.array([case[1] for case in cases])

        self.model.fit(features, targets, epochs=epochs, batch_size=5)

    def accuracy(self, cases):
        features = numpy.array([case[0] for case in cases])
        targets = numpy.array([case[1] for case in cases])

        scores = self.model.evaluate(features, targets)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

    def distribution(self, state):
        state = numpy.array(state)
        prediction = self.model.predict(state)
        #print(state, prediction)

        return prediction

    def normalize(self, vector):
        vector_sum = sum(vector)
        if vector_sum == 0:
            return list(vector)
        return [float(i)/vector_sum for i in vector]

    def load_model(self, file_name):
        try:
            loaded_model = load_model(file_name + '.h5')
            self.model = loaded_model
        except:
            pass

    def save_model(self, file_name):
        self.model.save(file_name + '.h5')
