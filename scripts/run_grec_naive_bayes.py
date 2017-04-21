__author__ = 'thiagocastroferreira'

import numpy as np
import utils

from parsers.ReferenceParser import ReferenceParser
from parsers.grec.Parser import GRECParser
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def fit(references, dfeatures = []):
    '''
    :param references: training set
    :param dfeatures:  features with discrete distribution
    :return: priori, discrete posteriori distribution
    '''

    # Calculate the priori
    priori = dict(map(lambda x: (x, 0.0), utils.type2id.keys()))
    del priori['other']

    for y in priori.keys():
        priori[y] = float(len(filter(lambda x: x['type'] == y, references))) / len(references)

    # Calculate the discrete posteriori distribution
    dposteriori = dict(map(lambda x: (x, dict(map(lambda x: (x, dict(map(lambda x: (x, 0.0), dfeatures[x]))), dfeatures.keys()))), priori.keys()))
    for y in dposteriori:
        for feature in dposteriori[y]:
            for value in dposteriori[y][feature]:
                prob = float(len(filter(lambda x: x['type'] == y and x[feature] == value, references))) / \
                       len(filter(lambda x: x['type'] == y, references))
                dposteriori[y][feature][value] = prob

    return priori, dposteriori

def run():
    parser = ReferenceParser(False)
    references = parser.run()
    del parser
    testset = GRECParser('../data/grec/oficial').run()

    results = {'actual':[], 'predicted': []}

    discrete_features = {'syncat': utils.syntax2id.keys(), \
                         # 'topic': utils.topic2id.keys(), \
                         'categorical-recency': utils.recency2id.keys(), \
                         # 'animacy': utils.animacy2id.keys(), \
                         'givenness': utils.givenness2id.keys(), \
                         'paragraph-givenness': utils.givenness2id.keys(), \
                         'sentence-givenness': utils.givenness2id.keys(), \
                         'pos-bigram': set(map(lambda x: x['pos-bigram'], testset))}
    # '+pos-bigram': set(map(lambda x: x['+pos-bigram'], references)), \
    # 'last-type': utils.type2id.keys(), \
    # 'genre': utils.genres2id.keys()}

    priori, dposteriori = fit(references, discrete_features)

    print 'TEST SET SIZE: ', len(testset)

    for test in testset:
        predictions = dict(map(lambda x: (x, 0.0), priori.keys()))
        for type in predictions:
            prob = priori[type]
            for feature in discrete_features.keys():
                prob = prob * dposteriori[type][feature][test[feature]]
            predictions[type] = prob
        sum = reduce(lambda x,y: x+y, predictions.values())
        if sum == 0:
            sum = 1

        for key in predictions:
            predictions[key] = round(predictions[key] / sum, 2)

        major_actual = test['type']
        major_predicted = filter(lambda x: predictions[x] == np.max(predictions.values()), predictions.keys())[0]

        # if major_actual == major_predicted:
        if test['reference-id'] != 1 and test['paragraph-recency'] != 0 and test['sentence-recency'] == 0:
            print 'Text: ', test['text-id'], 'Slot: ', test['reference-id']
            print 'Name: ', test['name']
            print 'Original: ', test['original-type']
            print 'Predictions: ', predictions
            print 'Features: ', dict(map(lambda x: (x, test[x]), discrete_features.keys()))
            print 20*'-'

        results['actual'].append(major_actual)
        results['predicted'].append(major_predicted)

    return results

if __name__ == '__main__':
    results = run()
    print '\n'
    print 'GENERAL:'
    print 20 * '-'
    print 'Accuracy argmax: ', accuracy_score(results['actual'], results['predicted'])

    print '\n'
    print classification_report(results['actual'], results['predicted'])