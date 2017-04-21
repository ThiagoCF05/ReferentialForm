__author__ = 'thiagocastroferreira'

import numpy as np
import utils
import measure

from parsers.ReferenceParser import ReferenceParser
from models.softmax_regression import SoftmaxRegression, SoftmaxMLP

from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import entropy

def parse(references, features, types):
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

        Y = {}
        for y in types:
            Y[y] = float(len(filter(lambda x: x['type'] == y, reference))) / len(reference)

        bclass = {'long':Y['name']+Y['description']+Y['demonstrative'], \
            'short':Y['pronoun']+Y['empty']}
        classes.append(Y)
        bclasses.append(bclass)

    return np.array(X, dtype='float32'), classes, bclasses

def run(binary = False):
    batch_size = 8

    parser = ReferenceParser()
    references = parser.run()
    # references = parser.parse_last_mention(references)
    # references = parser.parse_frequencies(references)
    references = parser.parse_ner(references)
    del parser

    # types = ['name', 'pronoun', 'description', 'demonstrative', 'empty']
    types = ['demonstrative', 'description', 'empty', 'name', 'pronoun']

    lists = list(set(map(lambda x: x['list-id'], references)))
    features = {'syncat': utils.syntax2id,\
                # 'topic': utils.topic2id, \
                'paragraph-recency': [],\
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

    results = {}
    y_actual, y_predicted = [], []

    for l in lists:
        print l
        results[l] = {'entropy_actual':[], 'entropy_predicted':[], \
                      'actual':[], 'predicted': [], \
                      'bcoeff': [], 'jsd':[], 'kld':[]}
        train = filter(lambda x: x['list-id'] != l, references)
        test = filter(lambda x: x['list-id'] == l, references)

        X_train, y_train_full, y_train_binary = parse(train, features, types)
        X_test, y_test_full, y_test_binary = parse(test, features, types)

        if binary:
            y_train = np.array(map(lambda x: x.values(), y_train_binary), dtype='float32')
            y_test = np.array(map(lambda x: x.values(), y_test_binary), dtype='float32')
        else:
            y_train = np.array(map(lambda x: x.values(), y_train_full), dtype='float32')
            y_test = np.array(map(lambda x: x.values(), y_test_full), dtype='float32')

        # soft = SoftmaxMLP(X_test.shape[1], int(3*X_test.shape[1]), y_test.shape[1])
        soft = SoftmaxRegression(X_test.shape[1], y_test.shape[1])

        lr = 0.01
        for e in range(15):
            X_batch, y_batch = [], []
            for i in range(X_train.shape[0]):
                X_batch.append(X_train[i])
                y_batch.append(y_train[i])

                if i % batch_size == 0:
                    soft.train(X_batch, y_batch, lr)
                    X_batch, y_batch = [], []

            # soft.train(X_train, y_train, lr)
            error = soft.validate(X_train, y_train)
            print 'Epoch ', e+1, 'Error: ', error

            lr = lr / (e+1)
        # w = soft.W.get_value()
        # print w.shape
        # for i in range(w.shape[0]):
        #     print features.keys()[i], w[i]).round(decimals=2)
        print '\n'

        distribution = soft.classify(X_test)

        y_actual.extend(np.argmax(y_test, axis=1))
        y_predicted.extend(np.argmax(distribution, axis=1))

        results[l]['actual'].extend(np.argmax(y_test, axis=1))
        results[l]['predicted'].extend(np.argmax(distribution, axis=1))

        print 'Accuracy argmax: ', accuracy_score(np.argmax(y_test, axis=1), np.argmax(distribution, axis=1))
        for y_i, y in enumerate(list(y_test)):
            predictions = map(lambda x: x, list(distribution[y_i]))
            actual = map(lambda x: x, list(y))

            bcoeff = round(measure.bhattacharya(actual, predictions), 2)
            jsd = round(measure.JSD(np.array(actual), np.array(predictions), 2), 2)
            kld = round(entropy(np.array(actual), np.array(predictions), 2), 2)

            results[l]['bcoeff'].append(bcoeff)
            results[l]['jsd'].append(jsd)
            results[l]['kld'].append(kld)

            # print 'Real: ', actual, '\t', 'Prediction: ', predictions
        # print '\n'
    return results

if __name__ == '__main__':
    results = run(False)

    actual, predicted = [], []
    bcoeffs, jsds, klds = [], [], []

    for l in results:
        print '\n'
        print l.upper()
        print 20 * '-'
        print 'Major Accuracy: ', accuracy_score(results[l]['actual'], results[l]['predicted'])
        print 'Bcoeff: ', np.mean(results[l]['bcoeff'])

        actual.extend(results[l]['actual'])
        predicted.extend(results[l]['predicted'])

        bcoeffs.extend(results[l]['bcoeff'])
        jsds.extend(results[l]['jsd'])
        klds.extend(results[l]['kld'])

    print '\n'
    print 'GENERAL:'
    print 20 * '-'
    print 'Major Accuracy: ', accuracy_score(actual, predicted)
    print 'Bcoeff: ', np.mean(bcoeffs)
    print 'JSD: ', np.mean(jsds)
    print 'KLD: ', np.mean(klds)

    print '\n'
    print classification_report(actual, predicted)