__author__ = 'thiagocastroferreira'

import numpy as np
import utils
import measure
import copy

from parsers.ReferenceParser import ReferenceParser
from parsers.grec.Parser import GRECParser

def parse_test(references, classes):
    '''
    :param references:
    :param classes:
    :return:
    '''
    test = {}
    keys = set(map(lambda x: (x['text-id'], x['reference-id']), references))

    for key in keys:
        test[key] = {}

        X = filter(lambda x: x['text-id'] == key[0] and x['reference-id'] == key[1], references)
        test[key]['X'] = X[0]

        Y = {}
        for y in classes:
            Y[y] = float(len(filter(lambda x: x['type'] == y, X))) / len(X)
        test[key]['y'] = Y
    return test

def fit(references):
    '''
    :param references: training set
    :return: priori, discrete posteriori distribution and continuous posteriori distribution
    '''

    # Calculate the priori
    priori = dict(map(lambda x: (x, 0.0), utils.type2id.keys()))
    del priori['other']

    for y in priori.keys():
        priori[y] = float(len(filter(lambda x: x['type'] == y, references))) / len(references)

    return priori

def probability(X, trainset, features, priori, gains):
    predictions = dict(map(lambda x: (x, 0.0), priori.keys()))

    for type in predictions:
        filtered = filter(lambda x: x['type'] == type, trainset)
        denominator = len(filtered)
        if denominator > 0:
            for feature in features:
                filtered = filter(lambda x: x[feature] == X[feature], filtered)

                if len(filtered) == 0: break

            numerator = len(filtered)
            predictions[type] = priori[type] * (float(numerator)/denominator)
        else:
            break

    sum = reduce(lambda x,y: x+y, predictions.values())
    if sum != 0:
        for key in predictions:
            predictions[key] = round(predictions[key] / sum, 2)
        return predictions
    else:
        dfeature = filter(lambda f: gains[f] == min(gains.values()), gains.keys())
        if len(dfeature) > 0:
            features.remove(dfeature[0])
            del gains[dfeature[0]]
        else:
            features = ['givenness', 'paragraph-givenness', 'sentence-givenness', 'syncat']
        return probability(X, trainset, features, priori, gains)

def run():
    parser = ReferenceParser(False, '../../data/xmls')
    testset = parser.run()
    del parser
    trainset = GRECParser('../../data/grec/oficial').run()

    results = {'actual':[], 'predicted': [], 'jsd':[], 'bcoeff':[]}

    features = {'syncat': utils.syntax2id.keys(), \
                # 'topic': utils.topic2id.keys(), \
                'categorical-recency': utils.recency2id.keys(), \
                # 'animacy': utils.animacy2id.keys(), \
                'givenness': utils.givenness2id.keys(), \
                'paragraph-givenness': utils.givenness2id.keys(), \
                'sentence-givenness': utils.givenness2id.keys(), \
                'pos-bigram': list(set(map(lambda x: x['pos-bigram'], trainset))), \
                '+pos-bigram': set(map(lambda x: x['+pos-bigram'], trainset))}
    # 'last-type': utils.type2id.keys(), \
    # 'genre': utils.genres2id.keys()}

    priori = fit(trainset)
    gains = measure.gain(trainset, features, priori.keys())

    bcoeffs = []
    test = parse_test(testset, priori.keys())

    for y in test:
        predictions = probability(test[y]['X'], trainset, features.keys(), priori, copy.copy(gains))

        bcoeff = round(np.sum(np.sqrt(np.array(test[y]['y'].values())) * np.sqrt(np.array(predictions.values()))), 2)
        jsd = round(measure.JSD(np.array(test[y]['y'].values()), np.array(predictions.values())), 2)
        print 'Text: ', y[0], 'Slot: ', y[1]
        print 'R: ', test[y]['y']
        print 'P: ', predictions
        print 'F: ', dict(map(lambda x: (x, test[y]['X'][x]), features.keys()))
        print 'Bcoeff: ', bcoeff
        print 'JSD: ', jsd
        print 20*'-'

        results['bcoeff'].append(bcoeff)
        results['jsd'].append(jsd)
    return results

if __name__ == '__main__':
    results = run()
    print 'BCoeff: ', np.mean(results['bcoeff'])
    print 'JSD: ', np.mean(results['jsd'])
