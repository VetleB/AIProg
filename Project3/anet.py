from keras.models import Sequential
from keras.layers import Dense
import numpy

class Anet:

    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(50, activation='relu', input_dim=34))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def train_on_rbuf_cases(self, cases):
        features = numpy.array([case[0] for case in cases])
        targets = numpy.array([case[1] for case in cases])
        print(features)
        self.model.fit(features, targets, epochs=1, batch_size=5)

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

        return [float(i)/vector_sum for i in vector]