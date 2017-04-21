__author__ = 'thiagocastroferreira'

import numpy as np
import utils

from parsers.ReferenceParser import ReferenceParser

from sklearn.metrics import mean_squared_error, classification_report
from sklearn.metrics import accuracy_score

import measure
import copy

def normalized_entropy(events):
    '''
    :param events:
    :return:
    '''
    entropy = 0.0
    for event in events:
        if events[event] != 0.0:
            entropy = entropy + ((events[event] * np.log(events[event])) / np.log(len(events.keys())))
    return -entropy

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
    parser = ReferenceParser(False)
    references = parser.run()
    references = parser.parse_last_mention(references)
    del parser

    lists = list(set(map(lambda x: x['list-id'], references)))

    results = {}
    for l in lists:
        results[l] = {'entropy_actual':[], 'entropy_predicted':[], \
                      'major_actual':[], 'major_predicted': [], \
                      'bcoeff': [], 'jsd':[], 'kld':[]}
        train = filter(lambda x: x['list-id'] != l, references)
        test = filter(lambda x: x['list-id'] == l, references)

        features = {'syncat': utils.syntax2id.keys(), \
                    # 'topic': utils.topic2id.keys(), \
                    'categorical-recency': utils.recency2id.keys(), \
                    'animacy': utils.animacy2id.keys(), \
                    'givenness': utils.givenness2id.keys(), \
                    'paragraph-givenness': utils.givenness2id.keys(), \
                    'sentence-givenness': utils.givenness2id.keys(), \
                    'pos-bigram': set(map(lambda x: x['pos-bigram'], references)), \
                    '+pos-bigram': set(map(lambda x: x['+pos-bigram'], references)), \
                    'pos-trigram': set(map(lambda x: x['pos-trigram'], references)), \
                    '+pos-trigram': set(map(lambda x: x['+pos-trigram'], references))}
                    # 'last-type': utils.type2id.keys(), \
                    # 'genre': utils.genres2id.keys()}

        priori = fit(train)
        gains = measure.gain(train, features, priori.keys())

        test = parse_test(test, priori.keys())

        for y in test:
            predictions = probability(test[y]['X'], train, features.keys(), priori, copy.copy(gains))

            print 'Text: ', y[0], 'Slot: ', y[1]
            print 'Original: ', test[y]['y']
            print 'Predictions: ', predictions
            print 20*'-'

            results[l]['entropy_actual'].append(normalized_entropy(test[y]['y']))
            results[l]['entropy_predicted'].append(normalized_entropy(predictions))

            results[l]['major_actual'].append(filter(lambda x: test[y]['y'][x] == np.max(test[y]['y'].values()), test[y]['y'].keys())[0])
            results[l]['major_predicted'].append(filter(lambda x: predictions[x] == np.max(predictions.values()), predictions.keys())[0])

            bcoeff = round(np.sum(np.sqrt(np.array(test[y]['y'].values())) * np.sqrt(np.array(predictions.values()))), 2)
            jsd = round(measure.JSD(np.array(test[y]['y'].values()), np.array(predictions.values())), 2)
            kld = round(measure.KLD(np.array(test[y]['y'].values()), np.array(predictions.values())), 2)

            results[l]['bcoeff'].append(bcoeff)
            results[l]['jsd'].append(jsd)
            results[l]['kld'].append(kld)

    return results

if __name__ == '__main__':
    results = run()

    entropy_actual, entropy_predicted = [], []
    major_actual, major_predicted = [], []
    bcoeffs, jsds, klds = [], [], []
    for l in results:
        # print '\n'
        # print l.upper()
        # print 20 * '-'
        # print 'Accuracy argmax: ', accuracy_score(results[l]['major_actual'], results[l]['major_predicted'])
        # print 'Bcoeff: ', np.mean(results[l]['bcoeff'])

        entropy_actual.extend(results[l]['entropy_actual'])
        entropy_predicted.extend(results[l]['entropy_predicted'])

        major_actual.extend(results[l]['major_actual'])
        major_predicted.extend(results[l]['major_predicted'])

        bcoeffs.extend(results[l]['bcoeff'])
        jsds.extend(results[l]['jsd'])
        klds.extend(results[l]['kld'])

    print '\n'
    print 'GENERAL:'
    print 20 * '-'
    print 'Accuracy argmax: ', accuracy_score(major_actual, major_predicted)
    print 'Bcoeff: ', np.mean(bcoeffs)
    print 'JSD: ', np.mean(jsds)
    print 'KLD: ', np.mean(klds)