__author__ = 'thiagocastroferreira'

import numpy as np
import utils
import measure
import sys
import copy

from parsers.ReferenceParser import ReferenceParser
from parsers.grec.Parser import GRECParser

from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import entropy

def fit(references, features = []):
    '''
    :param references: training set
    :param features:  features with discrete distribution
    :return: priori, discrete posteriori distribution
    '''

    # Calculate the priori
    priori = dict(map(lambda x: (x, 0.0), utils.type2id.keys()))
    del priori['other']

    for y in priori.keys():
        priori[y] = float(len(filter(lambda x: x['type'] == y, references))) / len(references)

    # Calculate the discrete posteriori distribution
    posteriori = dict(map(lambda x: (x, dict(map(lambda x: (x, dict(map(lambda x: (x, 0.0), features[x]))), features.keys()))), priori.keys()))
    for y in posteriori:
        for feature in posteriori[y]:
            for value in posteriori[y][feature]:
                denominator = len(filter(lambda x: x['type'] == y, references))
                if denominator > 0:
                    prob = float(len(filter(lambda x: x['type'] == y and x[feature] == value, references))) / \
                        denominator
                else:
                    prob = 0

                posteriori[y][feature][value] = prob

    return priori, posteriori

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

def predict(X, priori, posteriori, features):
    # print features
    predictions = dict(map(lambda x: (x, 0.0), priori.keys()))
    for type in predictions:
        prob = priori[type]
        for feature in features:
            try:
                prob = prob * posteriori[type][feature][X[feature]]
            except:
                print 'ERROR'
                prob = prob * 0.0
        predictions[type] = prob

    sum = reduce(lambda x,y: x+y, predictions.values())
    if sum == 0:
        if len(features) == 0:
            maxv = 1.0 - (4*sys.float_info.min)
            minv = sys.float_info.min
            predictions = {'name':maxv, 'pronoun':minv, 'description':minv, 'demonstrative':minv, 'empty':minv}
        else:
            del features[0]
            predictions = predict(X, priori, posteriori, features)
    else:
        sum = sum + (len(priori.keys()) * sys.float_info.min)
        for key in predictions:
            ''' SMOOTH THE DISTRIBUTION WITH THE MINIMUN FLOAT '''
            predictions[key] = (predictions[key]+sys.float_info.min) / sum
    return predictions

def run():
    parser = ReferenceParser(False, True, '../../data/xmls')
    testset = parser.run()
    del parser
    trainset = GRECParser('../../data/grec/oficial').run()

    results = {'major_actual':[], 'major_predicted': [], \
                  'binary_actual':[], 'binary_predicted':[], \
                  'bbcoeff': [], 'bjsd':[], 'bkld':[], \
                  'bcoeff': [], 'jsd':[], 'kld':[]}

    features = {'syncat': utils.syntax2id.keys(), \
                         # 'topic': utils.topic2id.keys(), \
                         'categorical-recency': utils.recency2id.keys(), \
                         # 'animacy': utils.animacy2id.keys(), \
                         'givenness': utils.givenness2id.keys(), \
                         'paragraph-givenness': utils.givenness2id.keys(), \
                         'sentence-givenness': utils.givenness2id.keys(),\
                         'pos-bigram': list(set(map(lambda x: x['pos-bigram'], trainset))),\
                         'pos-trigram': set(map(lambda x: x['pos-trigram'], trainset)), \
                         '+pos-trigram': set(map(lambda x: x['+pos-trigram'], trainset)), \
                         '+pos-bigram': set(map(lambda x: x['+pos-bigram'], trainset))}
    # 'last-type': utils.type2id.keys(), \
    # 'genre': utils.genres2id.keys()}
    # gain_order = ['+pos-trigram', 'pos-trigram', '+pos-bigram', 'pos-bigram', 'categorical-recency', 'syncat', \
    #               'givenness', 'sentence-givenness', 'paragraph-givenness']
    gain_order = ['categorical-recency', 'syncat', 'givenness', '+pos-trigram', '+pos-bigram', \
                  'pos-trigram', 'sentence-givenness', 'paragraph-givenness', 'pos-bigram']

    priori, posteriori = fit(trainset, features)

    test = parse_test(testset, priori.keys())

    for y in test:
        predictions = predict(test[y]['X'], priori, posteriori, copy.copy(gain_order))

        major_actual = filter(lambda x: test[y]['y'][x] == np.max(test[y]['y'].values()), test[y]['y'].keys())[0]
        major_predicted = filter(lambda x: predictions[x] == np.max(predictions.values()), predictions.keys())[0]

        bcoeff = round(measure.bhattacharya(test[y]['y'].values(), predictions.values()), 2)
        jsd = round(measure.JSD(np.array(test[y]['y'].values()), np.array(predictions.values()), 2), 2)
        kld = round(entropy(np.array(test[y]['y'].values()), np.array(predictions.values()), 2), 2)
        cross = round(measure.crossentropy(np.array(test[y]['y'].values()), np.array(predictions.values())), 2)
        # print 'Text: ', y[0], 'Slot: ', y[1]
        # print 'R: ', test[y]['y']
        # print 'P: ', predictions
        # print 'F: ', dict(map(lambda x: (x, test[y]['X'][x]), features.keys()))
        # print 'Bcoeff: ', bcoeff
        # print 'JSD: ', jsd
        # print 20*'-'

        bactual_distribution = { \
            'long':test[y]['y']['name']+test[y]['y']['description']+test[y]['y']['demonstrative'], \
            'short':test[y]['y']['pronoun']+test[y]['y']['empty']}
        bpredicted_distribution = { \
            'long':predictions['name']+predictions['description']+predictions['demonstrative'], \
            'short':predictions['pronoun']+predictions['empty']}

        bbcoeff = round(measure.bhattacharya(bactual_distribution.values(), bpredicted_distribution.values()), 2)
        bjsd = round(measure.JSD(np.array(bactual_distribution.values()), np.array(bpredicted_distribution.values()), 2), 2)
        bkld = round(entropy(np.array(bactual_distribution.values()), np.array(bpredicted_distribution.values())), 2)
        bcross = round(measure.crossentropy(np.array(bactual_distribution.values()), np.array(bpredicted_distribution.values())), 2)

        binary_actual = filter(lambda x: bactual_distribution[x] == np.max(bactual_distribution.values()), bactual_distribution.keys())[0]
        binary_predicted = filter(lambda x: bpredicted_distribution[x] == np.max(bpredicted_distribution.values()), bpredicted_distribution.keys())[0]

        results['major_actual'].append(major_actual)
        results['major_predicted'].append(major_predicted)

        results['binary_actual'].append(binary_actual)
        results['binary_predicted'].append(binary_predicted)

        results['bcoeff'].append(bcoeff)
        results['jsd'].append(jsd)
        results['kld'].append(kld)

        results['bbcoeff'].append(bbcoeff)
        results['bjsd'].append(bjsd)
        results['bkld'].append(bkld)
    return results

if __name__ == '__main__':
    results = run()
    print '\n'
    print 'GENERAL:'
    print 20 * '-'
    print 'Major Accuracy: ', accuracy_score(results['major_actual'], results['major_predicted'])
    print 'Bcoeff: ', np.mean(results['bcoeff'])
    print 'JSD: ', np.mean(results['jsd'])
    print 'KLD: ', np.mean(results['kld'])
    print 20 * '-'
    print 'Binary Accuracy: ', accuracy_score(results['binary_actual'], results['binary_predicted'])
    print 'Bcoeff: ', np.mean(results['bbcoeff'])
    print 'JSD: ', np.mean(results['bjsd'])
    print 'KLD: ', np.mean(results['bkld'])

    print '\n'
    print classification_report(results['major_actual'], results['major_predicted'])

    print '\n'
    print classification_report(results['binary_actual'], results['binary_predicted'])
