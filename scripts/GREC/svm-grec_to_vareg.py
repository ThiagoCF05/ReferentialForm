__author__ = 'thiagocastroferreira'

from parsers.ReferenceParser import ReferenceParser
from parsers.grec.Parser import GRECParser
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import entropy

import utils
import measure
import numpy as np

def parse_train(references, features, forms, type2bin):
    X, Y, Y_binary = [], [], []
    for reference in references:
        _x = []
        for f in features.keys():
            if features[f] == []:
                _x.append(reference[f])
            else:
                _x.append(features[f][reference[f]])
        X.append(_x)
        Y.append(forms.index(reference['type']))
        Y_binary.append(type2bin[reference['type']])

    X.append(_x)
    Y.append(0)
    Y_binary.append(type2bin['demonstrative'])

    return X, Y, Y_binary

def parse_test(references, features, types):
    '''
    :param references:
    :param classes:
    :return:
    '''
    X, classes, bclasses = [], [], []
    keys = set(map(lambda x: (x['text-id'], x['reference-id']), references))

    for key in keys:
        reference = filter(lambda x: x['text-id'] == key[0] and x['reference-id'] == key[1], references)
        x_ = []
        for f in features.keys():
            if features[f] == []:
                x_.append(reference[0][f])
            else:
                x_.append(features[f][reference[0][f]])
        X.append(x_)

        Y, Y_bin = [], [0.0, 0.0]
        for y in types:
            prob = float(len(filter(lambda x: x['type'] == y, reference))) / len(reference)
            Y.append(prob)

            if y in ['name', 'description', 'demonstrative']:
                Y_bin[0] += prob
            else:
                Y_bin[1] += prob
        classes.append(Y)
        bclasses.append(Y_bin)

    return np.array(X, dtype='float32'), classes, bclasses

def run(binary = False):
    parser = ReferenceParser(False, True, '../../data/xmls')
    testset = parser.run()
    # testset = parser.parse_frequencies(testset)
    testset = parser.parse_ner(testset)

    parser = GRECParser('../../data/grec/oficial')
    trainset = parser.run()
    # trainset = parser.parse_frequencies(trainset)
    trainset = parser.parse_ner(trainset)
    del parser

    type2bin = {'name':1, 'pronoun':0, 'description':1, 'demonstrative':1, 'empty':0}
    forms = ['demonstrative', 'description', 'empty', 'name', 'pronoun']

    features = {'syncat': utils.syntax2id, \
                # 'topic': utils.topic2id, \
                'paragraph-recency': [], \
                'paragraph-position': [], \
                'sentence-recency': [], \
                'sentence-position': [], \
                'num-entities':[], \
                'num-entities-same':[], \
                'dist-entities':[], \
                'dist-entities-same':[], \
                # 'categorical-recency': utils.recency2id, \
                # 'paragraph-id': [], \
                # 'sentence-id': [], \
                # 'reference-id':[],\
                # 'animacy': utils.animacy2id, \
                'givenness': utils.givenness2id, \
                'paragraph-givenness': utils.givenness2id, \
                'sentence-givenness': utils.givenness2id}
    # 'parallelism': utils.topic2id}
    # 'pos-bigram': dict([(x, xi) for xi, x in enumerate(set(map(lambda x: x['pos-bigram'], references)))]),\
    # '+pos-bigram': dict([(x, xi) for xi, x in enumerate(set(map(lambda x: x['+pos-bigram'], references)))]), \
    # 'pos-trigram': dict([(x, xi) for xi, x in enumerate(set(map(lambda x: x['pos-trigram'], references)))]), \
    # '+pos-trigram': dict([(x, xi) for xi, x in enumerate(set(map(lambda x: x['+pos-trigram'], references)))]),\
    # 'last-type': utils.type2id, \
    # 'competitor': utils.competitor2id, \
    # 'genre': utils.genres2id,\
    # 'freq-name':[],\
    # 'freq-pronoun':[],\
    # 'freq-description':[],\
    # 'freq-demonstrative':[],\
    # 'freq-empty':[]}

    results = {'entropy_actual':[], 'entropy_predicted':[], \
                  'actual':[], 'predicted': [], \
                  'bcoeff': [], 'jsd':[], 'kld':[]}

    X_train, y_train_full, y_train_binary = parse_train(trainset, features, forms, type2bin)
    X_test, y_test_full, y_test_binary = parse_test(testset, features, forms)

    if binary:
        y_train = np.array(y_train_binary, dtype='float32')
        y_test = np.array(y_test_binary, dtype='float32')
    else:
        y_train = np.array(y_train_full, dtype='float32')
        y_test = np.array(y_test_full, dtype='float32')

    clf = svm.SVC(probability=True)
    print len(X_train)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    distributions = clf.predict_proba(X_test)

    results['actual'].extend(np.argmax(y_test, axis=1))
    results['predicted'].extend(predictions)

    for y_i in range(len(y_test)):
        bcoeff = round(measure.bhattacharya(np.array(y_test[y_i]), np.array(distributions[y_i])), 2)
        jsd = round(measure.JSD(np.array(y_test[y_i]), np.array(distributions[y_i]), 2), 2)
        kld = round(entropy(np.array(y_test[y_i]), np.array(distributions[y_i]), 2), 2)

        results['bcoeff'].append(bcoeff)
        results['jsd'].append(jsd)
        results['kld'].append(kld)

    return results

if __name__ == '__main__':
    results = run(False)

    print '\n'
    print 'GENERAL:'
    print 20 * '-'
    print 'Major Accuracy: ', accuracy_score(results['actual'], results['predicted'])
    print 'Bcoeff: ', np.mean(results['bcoeff'])
    print 'JSD: ', np.mean(results['jsd'])
    print 'KLD: ', np.mean(results['kld'])

    print '\n'
    print classification_report(results['actual'], results['predicted'])