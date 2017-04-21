__author__ = 'thiagocastroferreira'

import numpy as np
import utils
import measure
import sys
import copy

from parsers.ReferenceParser import ReferenceParser

from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import entropy

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

def fit(references, features = []):
    '''
    :param references: training set
    :param features:  features with discrete distribution
    :param cfeatures:  features with continuous distribution
    :return: priori, discrete posteriori distribution and continuous posteriori distribution
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
                prob = float(len(filter(lambda x: x['type'] == y and x[feature] == value, references))) / \
                       len(filter(lambda x: x['type'] == y, references))
                posteriori[y][feature][value] = prob

    return priori, posteriori

def predict(X, priori, posteriori, features):
    # print features
    predictions = dict(map(lambda x: (x, 0.0), priori.keys()))
    for type in predictions:
        prob = priori[type]
        for feature in features:
            prob = prob * posteriori[type][feature][X[feature]]
        predictions[type] = prob

    sum = reduce(lambda x,y: x+y, predictions.values())
    if sum == 0:
        if len(features) == 0:
            maxv = 1.0 - ((len(priori.keys())-1)*sys.float_info.min)
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
    '''
    :entropy_actual -> real entropy
    :entropy_predicted -> predicted entropy

    :major_actual -> real major choice among the 5 referential forms
    :major_predicted -> predicted major choice among the 5 referential forms

    :binary_actual -> real major choice among long and short descriptions
    :binary_predicted -> predicted major choice among long and short descriptions

    :bcoeff, :jsd -> statistical distance measures for the 5 referential forms distribution

    :bjsd -> statistical distance measures for the long/short descriptions distribution
    '''
    parser = ReferenceParser()
    references = parser.run()
    del parser

    lists = list(set(map(lambda x: x['list-id'], references)))

    results = {}
    for l in lists:
        results[l] = {'entropy_actual':[], 'entropy_predicted':[], \
                      'major_actual':[], 'major_predicted': [], \
                      'binary_actual':[], 'binary_predicted':[], \
                      'bbcoeff': [], 'bjsd':[], 'bkld':[], \
                      'bcoeff': [], 'jsd':[], 'kld':[]}

        train = filter(lambda x: x['list-id'] != l, references)
        test = filter(lambda x: x['list-id'] == l, references)

        features = {'syncat': utils.syntax2id.keys(), \
                    'categorical-recency': utils.recency2id.keys(),\
                    # 'animacy': utils.animacy2id.keys(), \
                    'givenness': utils.givenness2id.keys(), \
                    'paragraph-givenness': utils.givenness2id.keys(), \
                    'sentence-givenness': utils.givenness2id.keys(),\
                    'pos-bigram': set(map(lambda x: x['pos-bigram'], references)),\
                    'pos-trigram': set(map(lambda x: x['pos-trigram'], references)),\
                    '+pos-bigram': set(map(lambda x: x['+pos-bigram'], references)),\
                    '+pos-trigram': set(map(lambda x: x['+pos-trigram'], references))}
        gain_order = ['categorical-recency', 'syncat', 'givenness', '+pos-trigram', '+pos-bigram',\
                      'pos-trigram', 'sentence-givenness', 'paragraph-givenness', 'pos-bigram']
        # gain_order = ['+pos-trigram', 'pos-trigram', '+pos-bigram', 'pos-bigram', 'categorical-recency', 'syncat', \
        #               'givenness', 'sentence-givenness', 'paragraph-givenness']

        priori, posteriori = fit(train, features)
        del train

        test = parse_test(test, priori.keys())

        for y in test:
            predictions = predict(test[y]['X'], priori, posteriori, copy.copy(gain_order))

            major_actual = filter(lambda x: test[y]['y'][x] == np.max(test[y]['y'].values()), test[y]['y'].keys())[0]
            major_predicted = filter(lambda x: predictions[x] == np.max(predictions.values()), predictions.keys())[0]

            bcoeff = round(measure.bhattacharya(test[y]['y'].values(), predictions.values()), 2)
            jsd = round(measure.JSD(np.array(test[y]['y'].values()), np.array(predictions.values()), 2), 2)
            kld = round(entropy(np.array(test[y]['y'].values()), np.array(predictions.values()), 2), 2)

            bactual_distribution = { \
                'long':test[y]['y']['name']+test[y]['y']['description']+test[y]['y']['demonstrative'], \
                'short':test[y]['y']['pronoun']+test[y]['y']['empty']}
            bpredicted_distribution = { \
                'long':predictions['name']+predictions['description']+predictions['demonstrative'], \
                'short':predictions['pronoun']+predictions['empty']}

            bbcoeff = round(measure.bhattacharya(bactual_distribution.values(), bpredicted_distribution.values()), 2)
            bjsd = round(measure.JSD(np.array(bactual_distribution.values()), np.array(bpredicted_distribution.values()), 2), 2)
            bkld = round(entropy(np.array(bactual_distribution.values()), np.array(bpredicted_distribution.values())), 2)

            binary_actual = filter(lambda x: bactual_distribution[x] == np.max(bactual_distribution.values()), bactual_distribution.keys())[0]
            binary_predicted = filter(lambda x: bpredicted_distribution[x] == np.max(bpredicted_distribution.values()), bpredicted_distribution.keys())[0]

            results[l]['entropy_actual'].append(normalized_entropy(test[y]['y']))
            results[l]['entropy_predicted'].append(normalized_entropy(predictions))

            results[l]['major_actual'].append(major_actual)
            results[l]['major_predicted'].append(major_predicted)

            results[l]['binary_actual'].append(binary_actual)
            results[l]['binary_predicted'].append(binary_predicted)

            results[l]['bcoeff'].append(bcoeff)
            results[l]['jsd'].append(jsd)
            results[l]['kld'].append(kld)

            results[l]['bbcoeff'].append(bbcoeff)
            results[l]['bjsd'].append(bjsd)
            results[l]['bkld'].append(bkld)

    return results

if __name__ == '__main__':
    results = run()

    entropy_actual, entropy_predicted = [], []
    major_actual, major_predicted = [], []
    binary_actual, binary_predicted = [], []
    bcoeffs, bbcoeffs = [], []
    jsds, klds = [], []
    bjsds, bklds = [], []

    for l in results:
        # print '\n'
        # print l.upper()
        # print 20 * '-'
        # print 'Major Accuracy: ', accuracy_score(results[l]['major_actual'], results[l]['major_predicted'])
        # print 'Binary Accuracy: ', accuracy_score(results[l]['binary_actual'], results[l]['binary_predicted'])
        # print 'Bcoeff: ', np.mean(results[l]['bcoeff'])

        entropy_actual.extend(results[l]['entropy_actual'])
        entropy_predicted.extend(results[l]['entropy_predicted'])

        major_actual.extend(results[l]['major_actual'])
        major_predicted.extend(results[l]['major_predicted'])

        binary_actual.extend(results[l]['binary_actual'])
        binary_predicted.extend(results[l]['binary_predicted'])

        bcoeffs.extend(results[l]['bcoeff'])
        jsds.extend(results[l]['jsd'])
        klds.extend(results[l]['kld'])

        bbcoeffs.extend(results[l]['bbcoeff'])
        bjsds.extend(results[l]['bjsd'])
        bklds.extend(results[l]['bkld'])

    print '\n'
    print 'GENERAL:'
    print 20 * '-'
    print 'Major Accuracy: ', accuracy_score(major_actual, major_predicted)
    print 'Bcoeff: ', np.mean(bcoeffs)
    print 'JSD: ', np.mean(jsds)
    print 'KLD: ', np.mean(klds)
    print 20 * '-'
    print 'Binary Accuracy: ', accuracy_score(binary_actual, binary_predicted)
    print 'Bcoeff: ', np.mean(bbcoeffs)
    print 'JSD: ', np.mean(bjsds)
    print 'KLD: ', np.mean(bklds)

    print '\n'
    print classification_report(major_actual, major_predicted)

    print '\n'
    print classification_report(binary_actual, binary_predicted)