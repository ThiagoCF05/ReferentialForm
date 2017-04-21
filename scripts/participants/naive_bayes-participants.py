__author__ = 'thiagocastroferreira'

from models.naive_bayes import NaiveBayes
from parsers.ReferenceParser import ReferenceParser

from scipy.stats import entropy, wilcoxon, spearmanr, pearsonr
from sklearn.metrics import classification_report, accuracy_score

import utils
import numpy as np
import measure

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
    parser = ReferenceParser(False, True, '../../data/xmls')
    references = parser.run()
    del parser

    features = {'syncat': utils.syntax2id.keys(), \
                # 'parallelism': ['false', 'true'], \
                # 'topic': utils.topic2id.keys(), \
                # 'categorical-recency': utils.recency2id.keys(),\
                # 'distractor': utils.distractor2id.keys(), \
                # 'previous': utils.distractor2id.keys()}
                # 'clause': utils.clause2id.keys(), \
                # 'competitor': utils.competitor2id.keys(), \
                # 'animacy': utils.animacy2id.keys(), \
                'givenness': utils.givenness2id.keys(), \
                'paragraph-givenness': utils.givenness2id.keys(), \
                'sentence-givenness': utils.givenness2id.keys()}
                # 'genre': utils.genres2id.keys()}

    forms = utils.type2id.keys()
    forms.remove('other')

    lists = list(set(map(lambda x: x['list-id'], references)))

    results = {}
    for l in lists:
        print l, '\r',
        results[l] = {'entropy_actual':[], 'entropy_predicted':[], \
                      'major_actual':[], 'major_predicted': [], \
                      'cross': [], 'bcross':[], \
                      'error_entropy': [], 'entropy': [], \
                      'bcoeff': [], 'jsd':[], 'kld':[]}

        participants = map(lambda x: x['participant-id'], references)

        for participant in participants:
            f = filter(lambda x: x['participant-id'] == participant, references)

            testset = filter(lambda x: x['file'] == 'news1', f)
            testset.extend(filter(lambda x: x['file'] == 'review1', f))
            testset.extend(filter(lambda x: x['file'] == 'wiki1', f))

            trainset = filter(lambda x: x['file'] != 'news1', f)
            trainset.extend(filter(lambda x: x['file'] != 'review1', f))
            trainset.extend(filter(lambda x: x['file'] != 'wiki1', f))

            bayes = NaiveBayes(trainset, features, forms)
            bayes.train()

            test = parse_test(testset, forms)

            for y in test:
                predictions = bayes.classify(test[y]['X'])

                major_actual = filter(lambda x: test[y]['y'][x] == np.max(test[y]['y'].values()), test[y]['y'].keys())[0]
                major_predicted = filter(lambda x: predictions[x] == np.max(predictions.values()), predictions.keys())[0]

                jsd = round(measure.JSD(np.array(test[y]['y'].values()), np.array(predictions.values()), 2), 2)
                cross = round(measure.crossentropy(np.array(test[y]['y'].values()), np.array(predictions.values())), 4)

                results[l]['entropy_actual'].append(entropy(test[y]['y'].values()))
                results[l]['entropy_predicted'].append(entropy(predictions.values()))

                results[l]['major_actual'].append(major_actual)
                results[l]['major_predicted'].append(major_predicted)

                results[l]['cross'].append(cross)

                # print 'Text: ', y[0], 'Slot: ', y[1]
                # print 'R: ', dict(map(lambda x: (x, round(test[y]['y'][x], 2)), test[y]['y'].keys()))
                # print 'P: ', dict(map(lambda x: (x, round(predictions[x], 2)), predictions.keys()))
                # print 'F: ', dict(map(lambda x: (x, test[y]['X'][x]), features.keys()))
                # print 'JSD: ', jsd
                # print 'Participant: ', participant
                # print 20*'-'

    return results

if __name__ == '__main__':
    results = run()

    entropy_actual, entropy_predicted = [], []
    major_actual, major_predicted = [], []
    binary_actual, binary_predicted = [], []
    rank_actual, rank_predicted = [], []
    cross, jsds = [], []
    entropy, error_entropy = [], []

    for l in results:
        entropy.extend(results[l]['entropy'])
        error_entropy.extend(results[l]['error_entropy'])

        entropy_actual.extend(results[l]['entropy_actual'])
        entropy_predicted.extend(results[l]['entropy_predicted'])

        major_actual.extend(results[l]['major_actual'])
        major_predicted.extend(results[l]['major_predicted'])

        jsds.extend(results[l]['jsd'])
        cross.extend(results[l]['cross'])

    print '\n'
    print 'GENERAL:'
    print 20 * '-'
    print 'Major Accuracy: ', accuracy_score(major_actual, major_predicted)
    print 'JSD: ', np.mean(jsds), np.std(jsds)
    print 'Cross-entropy: ', np.mean(cross)
    # print 'Error Entropy: ', np.mean(error_entropy), np.std(error_entropy), '/', np.mean(entropy), np.std(entropy)
    print 20 * '-'

    print '\n'
    print pearsonr(entropy_actual, entropy_predicted)

    print '\n'
    print classification_report(major_actual, major_predicted)