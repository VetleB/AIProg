import numpy as np
import numpy.random as NPR



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

        NPR.shuffle(ca)
        ca = ca[:round(self.case_fraction*len(ca))]
        self.cases = ca

        sep1 = round(len(ca)*self.test_fraction)
        self.test_cases = ca[0:sep1]

        sep2 = sep1 + round(len(ca)*self.validation_fraction)
        self.validation_cases = ca[sep1:sep2]

        self.train_cases = ca[sep2:]

    def get_training_cases(self): return self.train_cases

    def get_validation_cases(self): return self.validation_cases

    def get_testing_cases(self): return self.test_cases

    def get_mapping_cases(self, num):
        NPR.shuffle(self.cases)
        return self.cases[:num]
