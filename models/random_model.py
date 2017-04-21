__author__ = 'thiagocastroferreira'

from random import randint

import sys

class RandomModel(object):
    def __init__(self, classes):
        self.classes = classes

    def classify(self):
        predictions = dict(map(lambda x: (x, 0.0), self.classes))

        for type in predictions:
            prob = randint(0, 1000)
            ''' SMOOTH THE DISTRIBUTION WITH THE MINIMUN FLOAT '''
            predictions[type] = prob + sys.float_info.min
        sum = reduce(lambda x,y: x+y, predictions.values())

        for key in predictions:
            predictions[key] = predictions[key] / sum

        return predictions