__author__ = 'thiagocastroferreira'

import sys

'''
Naive bayes algorithm

WARNING: This code only handles discrete features
'''
class NaiveBayes(object):
    def __init__(self, trainset, features, classes):
        '''
        :param trainset: data to fit the probabilities
        :param features: discrete features that describe the data
        :param classes: classes that the data can assume
        :return:
        '''
        self.trainset = trainset
        self.features = features
        self.classes = classes

    def train(self):
        self.priori = self.__priori__()
        self.posteriori = self.__posteriori__()

    def __priori__(self):
        priori = dict(map(lambda x: (x, 0.0), self.classes))

        dem = len(filter(lambda x: x['type'] in self.classes, self.trainset))
        for y in priori:
            num = len(filter(lambda x: x['type'] == y, self.trainset))
            priori[y] = float(num) / dem
        return priori

    def __posteriori__(self):
        posteriori = dict(map(lambda x: (x, dict(map(lambda x: (x, dict(map(lambda x: (x, 0.0), self.features[x]))), self.features.keys()))), self.classes))
        for y in posteriori:
            for feature in posteriori[y]:
                for value in posteriori[y][feature]:
                    dem = len(filter(lambda x: x['type'] == y, self.trainset))
                    if dem > 0:
                        prob = float(len(filter(lambda x: x['type'] == y and x[feature] == value, self.trainset))) / dem
                    else:
                        prob = 0
                    posteriori[y][feature][value] = prob
        return posteriori

    def classify(self, X = {}, smooth='minimun_float'):
        predictions = dict(map(lambda x: (x, 0.0), self.classes))

        for type in predictions:
            prob = self.priori[type]
            for feature in self.features:
                prob = prob * self.posteriori[type][feature][X[feature]]

            predictions[type] = prob
            if smooth == 'minimun_float':
                ''' SMOOTH THE DISTRIBUTION WITH THE MINIMUN FLOAT '''
                predictions[type] += sys.float_info.min
        sum = reduce(lambda x,y: x+y, predictions.values())
        for key in predictions:
            predictions[key] = float(predictions[key]) / sum

        return predictions