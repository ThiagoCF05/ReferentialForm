__author__ = 'thiagocastroferreira'

import sys
sys.path.append('../../')
import numpy as np
import utils
import measure

from parsers.ReferenceParser import ReferenceParser
from parsers.grec.Parser import GRECParser

from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import entropy, spearmanr, rankdata

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

def run(features):
    parser = ReferenceParser(False, True, '../../data/xmls')
    testset = parser.run()
    testset = parser.parse_ner(testset)

    parser = GRECParser('../../data/grec/oficial')
    trainset = parser.run()
    # trainset = parser.parse_ner(trainset)
    del parser

    print "SIZE: ", len(trainset)

    genres = {'news':{'entropy_actual':[], 'entropy_predicted':[], \
                      'major_actual':[], 'major_predicted': [], 'original': [], \
                      'binary_actual':[], 'binary_predicted':[], \
                      'rank_actual':[], 'rank_predicted':[], \
                      'bbcoeff': [], 'bjsd':[], 'bkld':[], \
                      'error_entropy': [], 'entropy': [], \
                      'bcoeff': [], 'jsd':[], 'kld':[]}, \
              'review':{'entropy_actual':[], 'entropy_predicted':[], \
                        'major_actual':[], 'major_predicted': [], 'original': [], \
                        'binary_actual':[], 'binary_predicted':[], \
                        'rank_actual':[], 'rank_predicted':[], \
                        'bbcoeff': [], 'bjsd':[], 'bkld':[], \
                        'error_entropy': [], 'entropy': [], \
                        'bcoeff': [], 'jsd':[], 'kld':[]}, \
              'wiki':{'entropy_actual':[], 'entropy_predicted':[], \
                      'major_actual':[], 'major_predicted': [], 'original': [], \
                      'binary_actual':[], 'binary_predicted':[], \
                      'rank_actual':[], 'rank_predicted':[], \
                      'bbcoeff': [], 'bjsd':[], 'bkld':[], \
                      'error_entropy': [], 'entropy': [], \
                      'bcoeff': [], 'jsd':[], 'kld':[]}}

    results = {'entropy_actual':[], 'entropy_predicted':[], \
               'major_actual':[], 'major_predicted': [], 'original': [], \
               'binary_actual':[], 'binary_predicted':[], \
                  'bbcoeff': [], 'bjsd':[], 'bkld':[], \
                  'rank_actual':[], 'rank_predicted':[], \
                  'error_entropy': [], 'right_entropy':[], 'entropy': [], \
                  'error_features': [], 'right_features':[],\
                  'bcoeff': [], 'jsd':[], 'kld':[]}

    priori, posteriori = fit(trainset, features)

    test = parse_test(testset, priori.keys())

    for y in test:
        predictions = dict(map(lambda x: (x, 0.0), priori.keys()))
        non_smoothed_predictions = dict(map(lambda x: (x, 0.0), priori.keys()))
        for type in predictions:
            prob = priori[type]
            for feature in features.keys():
                try:
                    prob = prob * posteriori[type][feature][test[y]['X'][feature]]
                except:
                    'ERRO', feature, test[y]['X'][feature]
            non_smoothed_predictions[type] = prob
            ''' SMOOTH THE DISTRIBUTION WITH THE MINIMUN FLOAT '''
            predictions[type] = prob + sys.float_info.min
        sum = reduce(lambda x,y: x+y, predictions.values())

        for key in predictions:
            predictions[key] = predictions[key] / sum

        original = test[y]['X']['original-type']
        major_actual = filter(lambda x: test[y]['y'][x] == np.max(test[y]['y'].values()), test[y]['y'].keys())[0]
        major_predicted = filter(lambda x: predictions[x] == np.max(predictions.values()), predictions.keys())[0]

        rank_actual = rankdata(test[y]['y'].values())
        rank_predicted = rankdata(predictions.values())

        bcoeff = round(measure.bhattacharya(test[y]['y'].values(), predictions.values()), 2)
        jsd = measure.JSD(np.array(test[y]['y'].values()), np.array(predictions.values()), 2)
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

        results['original'].append(original)
        results['major_actual'].append(major_actual)
        results['major_predicted'].append(major_predicted)

        results['binary_actual'].append(binary_actual)
        results['binary_predicted'].append(binary_predicted)

        results['rank_actual'].append(rank_actual)
        results['rank_predicted'].append(rank_predicted)

        results['entropy_actual'].append(entropy(test[y]['y'].values()))
        results['entropy_predicted'].append(entropy(predictions.values()))

        results['bcoeff'].append(bcoeff)
        results['jsd'].append(jsd)
        results['kld'].append(kld)

        results['bbcoeff'].append(bbcoeff)
        results['bjsd'].append(bjsd)
        results['bkld'].append(bkld)

        results['entropy'].append(test[y]['X']['entropy'])

        genre = test[y]['X']['genre']

        genres[genre]['entropy_actual'].append(entropy(test[y]['y'].values()))
        genres[genre]['entropy_predicted'].append(entropy(predictions.values()))

        genres[genre]['original'].append(original)
        genres[genre]['major_actual'].append(major_actual)
        genres[genre]['major_predicted'].append(major_predicted)

        genres[genre]['binary_actual'].append(binary_actual)
        genres[genre]['binary_predicted'].append(binary_predicted)

        genres[genre]['rank_actual'].append(rank_actual)
        genres[genre]['rank_predicted'].append(rank_predicted)

        genres[genre]['bcoeff'].append(bcoeff)
        genres[genre]['jsd'].append(jsd)
        genres[genre]['kld'].append(kld)

        genres[genre]['bbcoeff'].append(bbcoeff)
        genres[genre]['bjsd'].append(bjsd)
        genres[genre]['bkld'].append(bkld)

        genres[genre]['entropy'].append(test[y]['X']['entropy'])

        if major_actual != major_predicted:
            results['error_entropy'].append(test[y]['X']['entropy'])
            results['error_features'].append(tuple(map(lambda x: test[y]['X'][x], features.keys())))
        else:
            results['right_entropy'].append(test[y]['X']['entropy'])
            results['right_features'].append(tuple(map(lambda x: test[y]['X'][x], features.keys())))
    return results, genres

if __name__ == '__main__':
    features = {
        'syncat': utils.syntax2id.keys(),
        # 'topic': utils.topic2id.keys(),
        'categorical-recency': utils.recency2id.keys(),
        # 'distractor': utils.distractor2id.keys(),
        # 'previous': utils.distractor2id.keys(),
        # 'clause': utils.clause2id.keys(),
        # 'animacy': utils.animacy2id.keys(),
        'givenness': utils.givenness2id.keys(),
        'paragraph-givenness': utils.givenness2id.keys(),
        'sentence-givenness': utils.givenness2id.keys(),
        # 'pos-bigram': set(map(lambda x: x['pos-bigram'], trainset)),
        # 'pos-trigram': set(map(lambda x: x['pos-trigram'], trainset)),
        # '+pos-trigram': set(map(lambda x: x['+pos-trigram'], trainset)),
        # '+pos-bigram': set(map(lambda x: x['+pos-bigram'], trainset)),
        # 'last-type': utils.type2id.keys(),
        # 'genre': utils.genres2id.keys()
    }
    results, genres = run(features)

    for genre in genres:
        print '\n'
        print genre
        print 20 * '-'
        print 'Major Accuracy: ', accuracy_score(genres[genre]['major_actual'], genres[genre]['major_predicted'])
        print 'Original Accuracy: ', accuracy_score(genres[genre]['original'], genres[genre]['major_predicted'])
        print 'Bcoeff: ', np.mean(genres[genre]['bcoeff'])
        print 'JSD: ', np.mean(genres[genre]['jsd']), np.std(genres[genre]['jsd'])
        print 'KLD: ', np.mean(genres[genre]['kld'])
        print 20 * '-'
        print 'Binary Accuracy: ', accuracy_score(genres[genre]['binary_actual'], genres[genre]['binary_predicted'])
        print 'Bcoeff: ', np.mean(genres[genre]['bbcoeff'])
        print 'JSD: ', np.mean(genres[genre]['bjsd'])
        print 'KLD: ', np.mean(genres[genre]['bkld'])
        print 20 * '-'

        # print '\n'
        # print classification_report(genres[genre]['major_actual'], genres[genre]['major_predicted'])
        #
        # print '\n'
        # print classification_report(genres[genre]['binary_actual'], genres[genre]['binary_predicted'])

    print '\n'
    print 'GENERAL:'
    print 20 * '-'
    print 'Major Accuracy: ', accuracy_score(results['major_actual'], results['major_predicted'])
    print 'Original Accuracy: ', accuracy_score(results['original'], results['major_predicted'])
    print 'Bcoeff: ', np.mean(results['bcoeff'])
    print 'JSD: ', np.mean(results['jsd'])
    print 'KLD: ', np.mean(results['kld'])
    print 'Error Entropy: ', np.mean(results['error_entropy']), np.std(results['error_entropy']), '/', np.mean(results['entropy']), np.std(results['entropy'])
    print 'Right Entropy: ', np.mean(results['right_entropy']), np.std(results['right_entropy']), '/', np.mean(results['entropy']), np.std(results['entropy'])
    print 20 * '-'
    print 'Binary Accuracy: ', accuracy_score(results['binary_actual'], results['binary_predicted'])
    print 'Bcoeff: ', np.mean(results['bbcoeff'])
    print 'JSD: ', np.mean(results['bjsd'])
    print 'KLD: ', np.mean(results['bkld'])
    print 20 * '-'
    rank = []
    for actual, predicted in zip(results['rank_actual'], results['rank_predicted']):
        aux = spearmanr(actual, predicted)[0]
        if str(aux) == 'nan':
            rank.append(0)
        else:
            rank.append(aux)
    print 'Rank: ', np.mean(rank)

    print '\n'
    print spearmanr(results['entropy_actual'], results['entropy_predicted'])

    # print '\n'
    # print classification_report(results['major_actual'], results['major_predicted'])

    # print '\n'
    # print classification_report(results['binary_actual'], results['binary_predicted'])

    # print '\n'
    # print 'Error Features: '
    # for features, count in Counter(results['error_features']).iteritems():
    #     print features, count
    #
    # print '\n'
    # print 'Right Features: '
    # for features, count in Counter(results['right_features']).iteritems():
    #     print features, count
