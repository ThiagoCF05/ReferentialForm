__author__ = 'thiagocastroferreira'

import numpy as np
import utils
import itertools
import sys
import measure

from parsers.ReferenceParser import ReferenceParser
from parsers.grec.Parser import GRECParser

from sklearn.metrics import classification_report, accuracy_score

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
    parser = ReferenceParser(False, True, '../../data/xmls')
    references = parser.run()
    del parser
    testset = GRECParser('../../data/grec/oficial').run()

    results = {'major_actual':[], 'major_predicted': [], \
               'binary_actual':[], 'binary_predicted':[], \
               'bbcoeff': [], 'bjsd':[], \
               'bcoeff':[], 'jsd': []}

    features = {'syncat': utils.syntax2id.keys(), \
                         # 'topic': utils.topic2id.keys(), \
                         'categorical-recency': utils.recency2id.keys(), \
                         # 'animacy': utils.animacy2id.keys(), \
                         'givenness': utils.givenness2id.keys(), \
                         'paragraph-givenness': utils.givenness2id.keys(), \
                         'sentence-givenness': utils.givenness2id.keys(),\
                         'pos-bigram': list(set(map(lambda x: x['pos-bigram'], testset))), \
                         '+pos-bigram': set(map(lambda x: x['+pos-bigram'], testset))}
    # 'last-type': utils.type2id.keys(), \
    # 'genre': utils.genres2id.keys()}

    priori, dposteriori = fit(references, features)

    combinations = list(itertools.product(*features.values()))

    for combination in combinations:
        predictions = dict(map(lambda x: (x, 0.0), priori.keys()))
        real = dict(map(lambda x: (x, 0.0), priori.keys()))
        for type in predictions:
            filtered = filter(lambda x: x['type'] == type, testset)
            prob = priori[type]
            for f_i, feature in enumerate(features.keys()):
                prob = prob * dposteriori[type][feature][combination[f_i]]
                filtered = filter(lambda x: x[feature] == combination[f_i], filtered)

            ''' SMOOTH THE DISTRIBUTION WITH THE MINIMUN FLOAT '''
            predictions[type] = prob + sys.float_info.min
            real[type] = len(filtered)

        sum = reduce(lambda x,y: x+y, predictions.values())

        for key in predictions:
            predictions[key] = predictions[key] / sum

        sum = reduce(lambda x,y: x+y, real.values())
        if sum > 0:
            for key in real:
                real[key] = float(real[key]) / sum

            major_actual = filter(lambda x: real[x] == np.max(real.values()), real.keys())[0]
            major_predicted = filter(lambda x: predictions[x] == np.max(predictions.values()), predictions.keys())[0]

            bcoeff = round(np.sum(np.sqrt(np.array(real.values())) * np.sqrt(np.array(predictions.values()))), 2)
            jsd = round(measure.JSD(np.array(real.values()), np.array(predictions.values()), 2), 2)

            bactual_distribution = {'long':real['name']+real['description']+real['demonstrative'], \
                'short':real['pronoun']+real['empty']}
            bpredicted_distribution = {'long':predictions['name']+predictions['description']+predictions['demonstrative'], \
                'short':predictions['pronoun']+predictions['empty']}

            bbcoeff = round(measure.bhattacharya(bactual_distribution.values(), bpredicted_distribution.values()), 2)
            bjsd = round(measure.JSD(np.array(bactual_distribution.values()), np.array(bpredicted_distribution.values()), 2), 2)

            binary_actual = filter(lambda x: bactual_distribution[x] == np.max(bactual_distribution.values()), bactual_distribution.keys())[0]
            binary_predicted = filter(lambda x: bpredicted_distribution[x] == np.max(bpredicted_distribution.values()), bpredicted_distribution.keys())[0]

            results['major_actual'].append(major_actual)
            results['major_predicted'].append(major_predicted)

            results['binary_actual'].append(binary_actual)
            results['binary_predicted'].append(binary_predicted)

            results['bcoeff'].append(bcoeff)
            results['jsd'].append(jsd)

            results['bbcoeff'].append(bbcoeff)
            results['bjsd'].append(bjsd)

            print 'Combination: ', dict(map(lambda x: (features.keys()[x], combination[x]), range(len(combination))))
            print 'R: ', real
            print 'P: ', predictions
            print 'JSD ', jsd
            print 20*'-'
    return results

if __name__ == '__main__':
    results = run()
    print '\n'
    print 'GENERAL:'
    print 20 * '-'
    print 'Major Accuracy: ', accuracy_score(results['major_actual'], results['major_predicted'])
    print 'Bcoeff: ', np.mean(results['bcoeff'])
    print 'JSD: ', np.mean(results['jsd'])
    print 20 * '-'
    print 'Binary Accuracy: ', accuracy_score(results['binary_actual'], results['binary_predicted'])
    print 'Bcoeff: ', np.mean(results['bbcoeff'])
    print 'JSD: ', np.mean(results['bjsd'])

    print '\n'
    print classification_report(results['major_actual'], results['major_predicted'])

    print '\n'
    print classification_report(results['binary_actual'], results['binary_predicted'])
