__author__ = 'thiagocastroferreira'

import numpy as np
import utils
import copy

from parsers.ReferenceParser import ReferenceParser
from parsers.grec.Parser import GRECParser
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import measure

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
    del parser
    testset = GRECParser('../data/grec/oficial').run()

    results = {'actual':[], 'predicted': []}

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

    priori = fit(references)
    gains = measure.gain(references, features, priori.keys())

    for test in testset:
        predictions = probability(test, references, features.keys(), priori, copy.copy(gains))

        major_actual = test['original-type']
        major_predicted = filter(lambda x: predictions[x] == np.max(predictions.values()), predictions.keys())[0]

        # if major_actual == major_predicted:
        if test['reference-id'] != 0 and test['paragraph-recency'] != 0 and test['sentence-recency'] == 0:
            print 'Text: ', test['text-id'], 'Slot: ', test['reference-id']
            print 'Original: ', test['original-type']
            print 'Predictions: ', predictions
            print 'Features: ', dict(map(lambda x: (x, test[x]), features))
            print 20*'-'

        results['actual'].append(major_actual)
        results['predicted'].append(major_predicted)

    return results

if __name__ == '__main__':
    results = run()
    # print '\n'
    # print 'GENERAL:'
    # print 20 * '-'
    # print 'Accuracy argmax: ', accuracy_score(results['actual'], results['predicted'])
    #
    # print '\n'
    # print classification_report(results['actual'], results['predicted'])