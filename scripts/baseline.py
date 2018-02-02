__author__ = 'thiagocastroferreira'

import sys
sys.path.append('../')
import numpy as np
import measure

from parsers.ReferenceParser import ReferenceParser

from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import entropy, spearmanr, rankdata

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

def run():
    parser = ReferenceParser(False)
    references = parser.run()
    del parser

    lists = list(set(map(lambda x: x['list-id'], references)))

    genres = {'news':{'entropy_actual':[], 'entropy_predicted':[], \
                      'major_actual':[], 'major_predicted': [], 'original': [], \
                      'binary_actual':[], 'binary_predicted':[], \
                      'rank_actual':[], 'rank_predicted':[], \
                      'cross': [], 'bcross':[], \
                      'bbcoeff': [], 'bjsd':[], 'bkld':[], \
                      'error_entropy': [], 'entropy': [], \
                      'bcoeff': [], 'jsd':[], 'kld':[]}, \
              'review':{'entropy_actual':[], 'entropy_predicted':[], \
                        'major_actual':[], 'major_predicted': [], 'original': [], \
                        'binary_actual':[], 'binary_predicted':[], \
                        'rank_actual':[], 'rank_predicted':[], \
                        'cross': [], 'bcross':[], \
                        'bbcoeff': [], 'bjsd':[], 'bkld':[], \
                        'error_entropy': [], 'entropy': [], \
                        'bcoeff': [], 'jsd':[], 'kld':[]}, \
              'wiki':{'entropy_actual':[], 'entropy_predicted':[], \
                      'major_actual':[], 'major_predicted': [], 'original': [], \
                      'binary_actual':[], 'binary_predicted':[], \
                      'rank_actual':[], 'rank_predicted':[], \
                      'cross': [], 'bcross':[], \
                      'bbcoeff': [], 'bjsd':[], 'bkld':[], \
                      'error_entropy': [], 'entropy': [], \
                      'bcoeff': [], 'jsd':[], 'kld':[]}}

    results = {}
    for l in lists:
        results[l] = {'entropy_actual':[], 'entropy_predicted':[], \
                      'major_actual':[], 'major_predicted': [], 'original': [], \
                      'binary_actual':[], 'binary_predicted':[], \
                      'rank_actual':[], 'rank_predicted':[], \
                      'cross': [], 'bcross':[], \
                      'bbcoeff': [], 'bjsd':[], 'bkld':[], \
                      'bcoeff': [], 'jsd':[], 'kld':[]}
        test = filter(lambda x: x['list-id'] == l, references)

        forms = ['name', 'pronoun', 'description', 'demonstrative', 'empty']
        test = parse_test(test, forms)

        for y in test:
            predictions = dict(map(lambda x: (x, 0.0), forms))
            if test[y]['X']['paragraph-givenness'] == 'new':
                predictions['name'] = 1.0
            else:
                predictions['pronoun'] = 1.0
            for type in predictions:
                ''' SMOOTH THE DISTRIBUTION WITH THE MINIMUN FLOAT '''
                predictions[type] = predictions[type] + sys.float_info.min
            sum = reduce(lambda x,y: x+y, predictions.values())

            for key in predictions:
                predictions[key] = predictions[key] / sum

            original = test[y]['X']['original-type']
            major_actual = filter(lambda x: test[y]['y'][x] == np.max(test[y]['y'].values()), test[y]['y'].keys())[0]
            major_predicted = filter(lambda x: predictions[x] == np.max(predictions.values()), predictions.keys())[0]

            bcoeff = round(measure.bhattacharya(test[y]['y'].values(), predictions.values()), 2)
            jsd = round(measure.JSD(np.array(test[y]['y'].values()), np.array(predictions.values()), 2), 2)
            kld = round(entropy(np.array(test[y]['y'].values()), np.array(predictions.values()), 2), 2)
            cross = round(measure.crossentropy(np.array(test[y]['y'].values()), np.array(predictions.values())), 4)

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

            print 'Text: ', y[0], 'Slot: ', y[1]
            print 'Original: ', test[y]['y']
            print 'Predictions: ', predictions
            print 'Bcoeff: ', bcoeff
            print 'JSD: ', jsd
            print 'KLD: ', kld
            print 20*'-'

            results[l]['entropy_actual'].append(normalized_entropy(test[y]['y']))
            results[l]['entropy_predicted'].append(normalized_entropy(predictions))

            results[l]['original'].append(original)
            results[l]['major_actual'].append(major_actual)
            results[l]['major_predicted'].append(major_predicted)

            results[l]['binary_actual'].append(binary_actual)
            results[l]['binary_predicted'].append(binary_predicted)

            results[l]['rank_actual'].append(rankdata(test[y]['y'].values()))
            results[l]['rank_predicted'].append(rankdata(predictions.values()))

            results[l]['cross'].append(cross)
            results[l]['bcross'].append(bcross)

            results[l]['bcoeff'].append(bcoeff)
            results[l]['jsd'].append(jsd)
            results[l]['kld'].append(kld)

            results[l]['bbcoeff'].append(bbcoeff)
            results[l]['bjsd'].append(bjsd)
            results[l]['bkld'].append(bkld)

            genre = test[y]['X']['genre']
            genres[genre]['entropy_actual'].append(entropy(test[y]['y'].values()))
            genres[genre]['entropy_predicted'].append(entropy(predictions.values()))

            genres[genre]['original'].append(original)
            genres[genre]['major_actual'].append(major_actual)
            genres[genre]['major_predicted'].append(major_predicted)

            genres[genre]['binary_actual'].append(binary_actual)
            genres[genre]['binary_predicted'].append(binary_predicted)

            genres[genre]['cross'].append(cross)
            genres[genre]['bcross'].append(bcross)

            genres[genre]['bcoeff'].append(bcoeff)
            genres[genre]['jsd'].append(jsd)
            genres[genre]['kld'].append(kld)

            genres[genre]['bbcoeff'].append(bbcoeff)
            genres[genre]['bjsd'].append(bjsd)
            genres[genre]['bkld'].append(bkld)

            genres[genre]['entropy'].append(test[y]['X']['entropy'])

    return results, genres

if __name__ == '__main__':
    results, genres = run()

    for genre in genres:
        print '\n'
        print genre
        print 20 * '-'
        print 'Major Accuracy: ', accuracy_score(genres[genre]['major_actual'], genres[genre]['major_predicted'])
        print 'Original Accuracy: ', accuracy_score(genres[genre]['original'], genres[genre]['major_predicted'])
        print 'Bcoeff: ', np.mean(genres[genre]['bcoeff'])
        print 'JSD: ', np.mean(genres[genre]['jsd']), np.std(genres[genre]['jsd'])
        print 'KLD: ', np.mean(genres[genre]['kld'])
        print 'Cross-entropy: ', np.mean(genres[genre]['cross'])
        print 20 * '-'
        print 'Binary Accuracy: ', accuracy_score(genres[genre]['binary_actual'], genres[genre]['binary_predicted'])
        print 'Bcoeff: ', np.mean(genres[genre]['bbcoeff'])
        print 'JSD: ', np.mean(genres[genre]['bjsd'])
        print 'KLD: ', np.mean(genres[genre]['bkld'])
        print 'Cross-entropy: ', np.mean(genres[genre]['bcross'])
        print 20 * '-'

        print '\n'
        print classification_report(genres[genre]['major_actual'], genres[genre]['major_predicted'])

        print '\n'
        print classification_report(genres[genre]['binary_actual'], genres[genre]['binary_predicted'])

    entropy_actual, entropy_predicted = [], []
    original, major_actual, major_predicted = [], [], []
    binary_actual, binary_predicted = [], []
    rank_actual, rank_predicted = [], []
    cross, bcross = [], []
    bcoeffs, bbcoeffs = [], []
    jsds, klds = [], []
    bjsds, bklds = [], []
    for l in results:
        print '\n'
        print l.upper()
        print 20 * '-'
        print 'Accuracy argmax: ', accuracy_score(results[l]['major_actual'], results[l]['major_predicted'])
        print 'Bcoeff: ', np.mean(results[l]['bcoeff'])

        entropy_actual.extend(results[l]['entropy_actual'])
        entropy_predicted.extend(results[l]['entropy_predicted'])

        original.extend(results[l]['original'])
        major_actual.extend(results[l]['major_actual'])
        major_predicted.extend(results[l]['major_predicted'])

        binary_actual.extend(results[l]['binary_actual'])
        binary_predicted.extend(results[l]['binary_predicted'])

        rank_actual.extend(results[l]['rank_actual'])
        rank_predicted.extend(results[l]['rank_predicted'])

        bcoeffs.extend(results[l]['bcoeff'])
        jsds.extend(results[l]['jsd'])
        klds.extend(results[l]['kld'])
        cross.extend(results[l]['cross'])

        bbcoeffs.extend(results[l]['bbcoeff'])
        bjsds.extend(results[l]['bjsd'])
        bklds.extend(results[l]['bkld'])
        bcross.extend(results[l]['bcross'])

    print '\n'
    print 'GENERAL:'
    print 20 * '-'
    print 'Major Accuracy: ', accuracy_score(major_actual, major_predicted)
    print 'Original Accuracy: ', accuracy_score(original, major_predicted)
    print 'Bcoeff: ', np.mean(bcoeffs)
    print 'JSD: ', np.mean(jsds)
    print 'KLD: ', np.mean(klds)
    print 'Cross-entropy: ', np.mean(cross)
    print 20 * '-'
    print 'Binary Accuracy: ', accuracy_score(binary_actual, binary_predicted)
    print 'Bcoeff: ', np.mean(bbcoeffs)
    print 'JSD: ', np.mean(bjsds)
    print 'KLD: ', np.mean(bklds)
    print 'Cross-entropy: ', np.mean(bcross)
    print 20 * '-'
    rank = [spearmanr(actual, predicted)[0] for actual, predicted in zip(rank_actual, rank_predicted)]
    print 'Rank: ', np.mean(rank)

    print '\n'
    print spearmanr(entropy_actual, entropy_predicted)

    print '\n'
    print classification_report(major_actual, major_predicted)

    print '\n'
    print classification_report(binary_actual, binary_predicted)