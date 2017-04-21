__author__ = 'thiagocastroferreira'

from parsers.ReferenceParser import ReferenceParser
from parsers.grec.Parser import GRECParser
from models.hmm.discrete.DiscreteHMM import DiscreteHMM

import numpy
import itertools
import utils
import measure
import sys

from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import entropy

class HMMGRECScript(object):
    '''
    T = length of the observation sequence
    N = number of hidden states
    M = number of observation symbols
    A = transition probabilities
    B = observation probabilities
    pi = initial probabilities
    '''

    def __init__(self, features, forms):
        parser = ReferenceParser(False, True, '../../data/xmls')
        self.references = parser.run()
        del parser

        self.forms = forms
        self.n = len(self.forms)
        self.features = features

        parser = GRECParser('../../data/grec/oficial')
        self.trainset = parser.run()
        # trainset = parser.parse_ner(trainset)

        parser = ReferenceParser(False, True, '../../data/xmls')
        self.testset = parser.run()
        del parser

    def init_pi(self):
        self.pi = numpy.zeros(self.n)

        f = filter(lambda x: x['paragraph-givenness'] == 'new', self.trainset)

        for i, form in enumerate(self.forms):
            self.pi[i] = float(len(filter(lambda x: x['type'] == self.forms[i], f))) / len(f)

    def init_A(self):
        self.A = numpy.zeros((self.n, self.n))

        # IN REVIEW
        for i in range(self.n):
            dem = len(filter(lambda x: x['type'] == self.forms[i], self.trainset))
            for j in range(self.n):
                num = 0
                z = filter(lambda x: x['type'] == self.forms[j], self.trainset)
                for reference in z:
                    g = filter(lambda x: x['text-id'] == reference['text-id'] \
                                         and int(x['reference-id']) == int(reference['reference-id'])-1 \
                                         and x['type'] == self.forms[i], self.trainset)
                    num = num + len(g)
                if dem == 0:
                    self.A[i][j] = 0
                else:
                    self.A[i][j] = float(num) / dem

            ''' SMOOTH DISTRIBUTION '''
            self.A[i] = self.A[i] + sys.float_info.min
            s = sum(self.A[i])

            for j, e in enumerate(self.A[i]):
                self.A[i][j] = self.A[i][j] / s

    def init_B(self):
        self.v = list(set(list(itertools.product(*self.features.values()))))
        self.v = [dict([(self.features.keys()[i], e) for i, e in enumerate(combination)]) for combination in self.v]
        self.m = len(self.v)
        self.B = numpy.zeros((self.n, self.m))

        for i, form in enumerate(self.forms):
            f = filter(lambda x: x['type'] == form, self.trainset)
            dem = len(f)
            for j, observation in enumerate(self.v):
                f = filter(lambda x: x['type'] == form, self.trainset)
                for feature in observation:
                    f = filter(lambda x: x[feature] == observation[feature], f)
                num = len(f)
                if dem > 0:
                    self.B[i][j] = float(num) / dem
                else:
                    self.B[i][j] = 0.0

            ''' SMOOTH DISTRIBUTION '''
            self.B[i] = self.B[i] + sys.float_info.min
            s = sum(self.B[i])

            for j, e in enumerate(self.B[i]):
                self.B[i][j] = self.B[i][j] / s

    def parse(self, _set):
        test = {}
        keys = set(map(lambda x: (x['text-id'], x['reference-id']), _set))

        for key in keys:
            test[key] = {}

            X = filter(lambda x: x['text-id'] == key[0] and x['reference-id'] == key[1], _set)
            test[key]['X'] = X[0]

            Y = {}
            for y in self.forms:
                Y[y] = float(len(filter(lambda x: x['type'] == y, X))) / len(X)
            test[key]['y'] = Y

        texts = set(map(lambda x: x['text-id'], _set))

        X, y = [], []
        for text in texts:
            f = filter(lambda x: x['text-id'] == text, _set)
            paragraphs = set(map(lambda x: x['paragraph-id'], f))

            for p in paragraphs:
                X_p, y_p = [], []
                g = filter(lambda x: x['paragraph-id'] == p, f)
                references = list(set(map(lambda x: int(x['reference-id']), g)))
                references.sort()

                for reference in references:
                    _x = dict([(x, test[(text, reference)]['X'][x]) for x in test[(text, reference)]['X'] if x in self.features.keys()])
                    _y = test[(text, reference)]['y']

                    X_p.append(self.v.index(_x))
                    y_p.append(_y)

                X.append(X_p)
                y.append(y_p)

        return X, y

    def run(self):
        results = {'major_actual':[], 'major_predicted': [], \
                      'bcoeff': [], 'jsd':[], 'kld':[]}

        self.init_pi()
        self.init_B()
        self.init_A()

        hmm2 = DiscreteHMM(self.n,self.m,self.A,self.B,self.pi,init_type='user',precision=numpy.float,verbose=True,scaled=True)

        X, y = self.parse(self.testset)

        for i in range(len(X)):
            hmm2._mapB(X[i])
            stats = hmm2._calcstats(X[i])
            gamma = stats['gamma']

            for j in range(len(X[i])):
                predictions = dict(zip(self.forms, gamma[j]))
                for type in predictions:
                    ''' SMOOTH THE DISTRIBUTION WITH THE MINIMUN FLOAT '''
                    predictions[type] = predictions[type] + sys.float_info.min
                sum = reduce(lambda x,y: x+y, predictions.values())

                for key in predictions:
                    predictions[key] = predictions[key] / sum

                # print y[i][j]
                # print predictions
                # print 10 * '-'

                major_actual = filter(lambda x: y[i][j][x] == numpy.max(y[i][j].values()), y[i][j].keys())[0]
                major_predicted = filter(lambda x: predictions[x] == numpy.max(predictions.values()), predictions.keys())[0]

                bcoeff = round(measure.bhattacharya(y[i][j].values(), predictions.values()), 2)
                jsd = round(measure.JSD(numpy.array(y[i][j].values()), numpy.array(predictions.values()), 2), 2)
                kld = round(entropy(numpy.array(y[i][j].values()), numpy.array(predictions.values()), 2), 2)

                results['major_actual'].append(major_actual)
                results['major_predicted'].append(major_predicted)

                results['bcoeff'].append(bcoeff)
                results['jsd'].append(jsd)
                results['kld'].append(kld)
        return results

if __name__ == '__main__':
    features = {'syncat': utils.syntax2id.keys(),
                'categorical-recency': utils.recency2id.keys(), \
                'givenness': utils.givenness2id.keys(), \
                'paragraph-givenness': utils.givenness2id.keys(), \
                'sentence-givenness': utils.givenness2id.keys()}
    # forms = ['name', 'pronoun', 'description', 'demonstrative', 'empty']
    forms = ['demonstrative', 'description', 'empty', 'name', 'pronoun']

    hmm = HMMGRECScript(features, forms)
    results = hmm.run()

    print '\n'
    print 'GENERAL:'
    print 20 * '-'
    print 'Major Accuracy: ', accuracy_score(results['major_actual'], results['major_predicted'])
    print 'Bcoeff: ', numpy.mean(results['bcoeff'])
    print 'JSD: ', numpy.mean(results['jsd']), numpy.std(results['jsd'])
    print 'KLD: ', numpy.mean(results['kld'])
    print 20 * '-'

    print '\n'
    print classification_report(results['major_actual'], results['major_predicted'])