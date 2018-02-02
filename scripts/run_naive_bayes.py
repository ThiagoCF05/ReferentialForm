__author__ = 'thiagocastroferreira'

import sys
sys.path.append('../')
import numpy as np
import utils
import measure

from parsers.ReferenceParser import ReferenceParser

from sklearn.metrics import classification_report, accuracy_score
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

def fit(references, dfeatures = [], cfeatures = []):
    '''
    :param references: training set
    :param dfeatures:  features with discrete distribution
    :param cfeatures:  features with continuous distribution
    :return: priori, discrete posteriori distribution and continuous posteriori distribution
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

    # Calculate the continuous posteriori distribution
    cposteriori = dict(map(lambda x: (x, dict(map(lambda x: (x, {'mean': 0.0, 'std': 0.0}), cfeatures))), priori.keys()))
    for y in cposteriori:
        for feature in cposteriori[y]:
            f = map(lambda x: x['recency'], filter(lambda x: x['type'] == y, references))
            cposteriori[y][feature]['mean'] = np.mean(f)
            cposteriori[y][feature]['std'] = np.std(f)

    return priori, dposteriori, cposteriori

def run(discrete_features, continuous_features):
    '''
    :entropy_actual -> real entropy
    :entropy_predicted -> predicted entropy

    :major_actual -> real major choice among the 5 referential forms
    :major_predicted -> predicted major choice among the 5 referential forms

    :binary_actual -> real major choice among long and short descriptions
    :binary_predicted -> predicted major choice among long and short descriptions

    :bcoeff, :jsd, :kld -> statistical distance measures for the 5 referential forms distribution

    :bjsd, :bkld -> statistical distance measures for the long/short descriptions distribution
    '''
    parser = ReferenceParser()
    references = parser.run()
    references = parser.parse_ner(references)
    del parser

    lists = list(set(map(lambda x: x['list-id'], references)))

    genres = {'news':{'entropy_actual':[], 'entropy_predicted':[], \
                      'major_actual':[], 'major_predicted': [], 'original': [], \
                      'binary_actual':[], 'binary_predicted':[], \
                      'rank_actual':[], 'rank_predicted':[], \
                      'cross': [], 'bcross':[], \
                      'bbcoeff': [], 'bjsd':[], 'bkld':[], \
                      'error_entropy': [], 'entropy': [], \
                      'bcoeff': [], 'jsd':[], 'kld':[]},\
              'review':{'entropy_actual':[], 'entropy_predicted':[], \
                        'major_actual':[], 'major_predicted': [], 'original': [], \
                        'binary_actual':[], 'binary_predicted':[], \
                        'rank_actual':[], 'rank_predicted':[], \
                        'cross': [], 'bcross':[], \
                        'bbcoeff': [], 'bjsd':[], 'bkld':[], \
                        'error_entropy': [], 'entropy': [], \
                        'bcoeff': [], 'jsd':[], 'kld':[]},\
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
                      'error_entropy': [], 'entropy': [], \
                      'bcoeff': [], 'jsd':[], 'kld':[]}

        train = filter(lambda x: x['list-id'] != l, references)
        test = filter(lambda x: x['list-id'] == l, references)

        priori, dposteriori, cposteriori = fit(train, discrete_features, continuous_features)
        del train

        test = parse_test(test, priori.keys())

        gaussian = lambda x, mean, std: (float(1) / (std * np.sqrt(2*np.pi))) * np.exp(-(((x-mean)**2) / (2 * std**2)))

        for y in test:
            predictions = dict(map(lambda x: (x, 0.0), priori.keys()))
            non_smoothed_predictions = dict(map(lambda x: (x, 0.0), priori.keys()))
            for type in predictions:
                prob = priori[type]
                for feature in discrete_features.keys():
                    prob = prob * dposteriori[type][feature][test[y]['X'][feature]]

                for feature in continuous_features:
                    aux = gaussian(test[y]['X'][feature], cposteriori[type][feature]['mean'], cposteriori[type][feature]['std'])
                    prob = prob * aux
                non_smoothed_predictions[type] = prob
                ''' SMOOTH THE DISTRIBUTION WITH THE MINIMUN FLOAT '''
                predictions[type] = prob + sys.float_info.min
            sum = reduce(lambda x,y: x+y, predictions.values())

            for key in predictions:
                predictions[key] = predictions[key] / sum
                non_smoothed_predictions[key] = non_smoothed_predictions[key] / sum


            original = test[y]['X']['original-type']
            major_actual = filter(lambda x: test[y]['y'][x] == np.max(test[y]['y'].values()), test[y]['y'].keys())[0]
            major_predicted = filter(lambda x: predictions[x] == np.max(predictions.values()), predictions.keys())[0]

            bcoeff = round(measure.bhattacharya(test[y]['y'].values(), predictions.values()), 2)
            jsd = measure.JSD(np.array(test[y]['y'].values()), np.array(predictions.values()), 2)
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

            results[l]['entropy_actual'].append(entropy(test[y]['y'].values()))
            results[l]['entropy_predicted'].append(entropy(predictions.values()))

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

            results[l]['entropy'].append(test[y]['X']['entropy'])

            genre = test[y]['X']['genre']
            genres[genre]['entropy_actual'].append(entropy(test[y]['y'].values()))
            genres[genre]['entropy_predicted'].append(entropy(predictions.values()))

            genres[genre]['original'].append(original)
            genres[genre]['major_actual'].append(major_actual)
            genres[genre]['major_predicted'].append(major_predicted)

            genres[genre]['binary_actual'].append(binary_actual)
            genres[genre]['binary_predicted'].append(binary_predicted)

            genres[genre]['rank_actual'].append(rankdata(test[y]['y'].values()))
            genres[genre]['rank_predicted'].append(rankdata(predictions.values()))

            genres[genre]['cross'].append(cross)
            genres[genre]['bcross'].append(bcross)

            genres[genre]['bcoeff'].append(bcoeff)
            genres[genre]['jsd'].append(jsd)
            genres[genre]['kld'].append(kld)

            genres[genre]['bbcoeff'].append(bbcoeff)
            genres[genre]['bjsd'].append(bjsd)
            genres[genre]['bkld'].append(bkld)

            genres[genre]['entropy'].append(test[y]['X']['entropy'])

            # if major_actual == major_predicted:
            #     results[l]['error_entropy'].append(test[y]['X']['entropy'])
            # print 'Text: ', y[0], 'Slot: ', y[1]
            # print 'R: ', dict(map(lambda x: (x, test[y]['y'][x]), test[y]['y'].keys()))
            # print 'P: ', dict(map(lambda x: (x, predictions[x]), predictions.keys()))
            # print 'F: ', dict(map(lambda x: (x, test[y]['X'][x]), discrete_features.keys()))
            # print 'JSD: ', jsd
            # print 20*'-'

            # if binary_actual != binary_predicted:
            #     print 'Text: ', y[0], 'Slot: ', y[1]
            #     print 'R: ', bactual_distribution
            #     print 'P: ', bpredicted_distribution
            #     print 'F: ', dict(map(lambda x: (x, test[y]['X'][x]), discrete_features.keys()))
            #     print 'JSD: ', bjsd
            #     print 20*'-'

    return results, genres

if __name__ == '__main__':
    discrete_features = {
        'syncat': utils.syntax2id.keys(),
        # 'parallelism': ['false', 'true'],
        # 'topic': utils.topic2id.keys(),
        'categorical-recency': utils.recency2id.keys(),
        # 'distractor': utils.distractor2id.keys(),
        # 'previous': utils.distractor2id.keys(),
        # 'clause': utils.clause2id.keys(),
        # 'competitor': utils.competitor2id.keys(),
        # 'animacy': utils.animacy2id.keys(),
        'givenness': utils.givenness2id.keys(),
        'paragraph-givenness': utils.givenness2id.keys(),
        'sentence-givenness': utils.givenness2id.keys()
    }
    # 'participant-id':[],\
    # 'pos-bigram': set(map(lambda x: x['pos-bigram'], references))}
    # 'pos-trigram': set(map(lambda x: x['pos-trigram'], references)), \
    # '+pos-bigram': set(map(lambda x: x['+pos-bigram'], references)), \
    # '+pos-trigram': set(map(lambda x: x['+pos-trigram'], references))}
    # 'last-type': utils.type2id.keys(),\
    # 'genre': utils.genres2id.keys()}

    results, genres = run(discrete_features, [])

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
        print 'Cross-entropy: ', np.mean(genres[genre]['bcross'])
        print 20 * '-'

        print '\n'
        print spearmanr(genres[genre]['entropy_actual'], genres[genre]['entropy_predicted'])

        # print '\n'
        # print classification_report(genres[genre]['major_actual'], genres[genre]['major_predicted'])
        #
        # print '\n'
        # print classification_report(genres[genre]['binary_actual'], genres[genre]['binary_predicted'])


    entropy_actual, entropy_predicted = [], []
    original, major_actual, major_predicted = [], [], []
    binary_actual, binary_predicted = [], []
    rank_actual, rank_predicted = [], []
    cross, bcross = [], []
    bcoeffs, bbcoeffs = [], []
    jsds, klds = [], []
    bjsds, bklds = [], []
    entropy, error_entropy = [], []

    for l in results:
        # print '\n'
        # print l.upper()
        # print 20 * '-'
        # print 'Major Accuracy: ', accuracy_score(results[l]['major_actual'], results[l]['major_predicted'])
        # print 'Binary Accuracy: ', accuracy_score(results[l]['binary_actual'], results[l]['binary_predicted'])
        # print 'Bcoeff: ', np.mean(results[l]['bcoeff'])

        entropy.extend(results[l]['entropy'])
        error_entropy.extend(results[l]['error_entropy'])

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
    print 'JSD: ', np.mean(jsds), np.std(jsds)
    print 'KLD: ', np.mean(klds)
    print 'Cross-entropy: ', np.mean(cross)
    # print 'Error Entropy: ', np.mean(error_entropy), np.std(error_entropy), '/', np.mean(entropy), np.std(entropy)
    print 20 * '-'
    print 'Binary Accuracy: ', accuracy_score(binary_actual, binary_predicted)
    print 'Bcoeff: ', np.mean(bbcoeffs)
    print 'JSD: ', np.mean(bjsds)
    print 'KLD: ', np.mean(bklds)
    print 'Cross-entropy: ', np.mean(bcross)
    print 20 * '-'
    rank = []
    for actual, predicted in zip(rank_actual, rank_predicted):
        aux = spearmanr(actual, predicted)[0]
        if str(aux) == 'nan':
            rank.append(0)
        else:
            rank.append(aux)
    print 'Rank: ', np.mean(rank)

    print '\n'
    print spearmanr(entropy_actual, entropy_predicted)

    # print '\n'
    # print classification_report(major_actual, major_predicted)
    #
    # print '\n'
    # print classification_report(binary_actual, binary_predicted)