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

        self.model.fit(features, targets, epochs=10, batch_size=5)
