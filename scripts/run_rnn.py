__author__ = 'thiagocastroferreira'

from parsers.ReferenceParser import ReferenceParser
from sklearn.metrics import classification_report, accuracy_score
from models.elman import ElmanRNN
from scipy.stats import entropy, spearmanr, rankdata

import utils
import itertools
import measure
import numpy
import time
import sys

class RNNScript(object):
    def __init__(self, features, forms):
        self.features = features
        self.forms = forms
        self.n = len(self.forms)

        parser = ReferenceParser()
        self.references = parser.run()
        self.references = parser.parse_ner(self.references)
        del parser

        self.reference2id = list(set(list(itertools.product(*self.features.values()))))
        self.reference2id = [dict([(self.features.keys()[i], e) for i, e in enumerate(combination)]) for combination in self.reference2id]
        self.reference2id = dict([(tuple(reference.values()), id) for id, reference in enumerate(self.reference2id)])

    def run(self, nh, de, cs, bs, lr, epochs):
        lists = list(set(map(lambda x: x['list-id'], self.references)))

        genres = {'news':{'major_actual':[], 'major_predicted': [], 'original': [], 'jsd':[]}, \
                  'review':{'major_actual':[], 'major_predicted': [], 'original': [], 'jsd': []}, \
                  'wiki':{'major_actual':[], 'major_predicted': [], 'original': [], 'jsd':[]}}
        results = {}
        for l in lists:
            results[l] = {'entropy_actual':[], 'entropy_predicted':[], \
                          'major_actual':[], 'major_predicted': [], 'original': [], \
                          'binary_actual':[], 'binary_predicted':[], \
                          'rank_actual':[], 'rank_predicted':[], \
                          'bcoeff': [], 'jsd':[], 'kld':[]}

            trainset = filter(lambda x: x['list-id'] != l, self.references)
            testset = filter(lambda x: x['list-id'] == l, self.references)

            self.train = self.parse(trainset)
            self.test = self.parse(testset)

            rnn = ElmanRNN(nh = nh,
                           nc = self.n,
                           ne = len(self.reference2id.keys()),
                           de = de,
                           cs = cs)

            # train with early stopping on validation set
            nsentences = len(self.train)
            for e in xrange(epochs):
                # shuffle
                # shuffle([train_lex, train_y], s['seed'])
                tic = time.time()
                for i in xrange(nsentences):
                    cwords = self.contextwin(self.train[i]['X'], cs)
                    words = map(lambda x: numpy.asarray(x).astype('int32'), \
                                self.minibatch(cwords, bs))
                    labels = self.train[i]['y']
                    for word_batch , label_last_word in zip(words, labels):
                        rnn.train(word_batch, label_last_word, lr)
                        rnn.normalize()
                    print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                    sys.stdout.flush()
                # lr = lr / (e+1)

            # evaluation // back into the real world : idx -> words
            predictions = [rnn.classify(numpy.asarray(self.contextwin(x['X'], cs)).astype('int32')) \
                                for x in self.test ]

            for i, x in enumerate(self.test):
                major_actual = numpy.argmax(x['y'], axis=1)
                major_predicted = numpy.argmax(predictions[i], axis=1)

                # print major_actual, major_predicted
                results[l]['original'].extend(x['original'])
                results[l]['major_actual'].extend(major_actual)
                results[l]['major_predicted'].extend(major_predicted)

                genre = x['genre']
                genres[genre]['original'].extend(x['original'])
                genres[genre]['major_actual'].extend(major_actual)
                genres[genre]['major_predicted'].extend(major_predicted)

                for j, gap in enumerate(x['y']):
                    bcoeff = round(measure.bhattacharya(numpy.array(x['y'][j]), numpy.array(predictions[i][j])), 2)
                    jsd = measure.JSD(numpy.array(x['y'][j]), numpy.array(predictions[i][j]), 2)
                    kld = round(entropy(numpy.array(x['y'][j]), numpy.array(predictions[i][j]), 2), 2)

                    results[l]['entropy_actual'].append(entropy(x['y'][j]))
                    results[l]['entropy_predicted'].append(entropy(predictions[i][j]))

                    results[l]['rank_actual'].append(rankdata(x['y'][j]))
                    results[l]['rank_predicted'].append(rankdata(predictions[i][j]))

                    results[l]['bcoeff'].append(bcoeff)
                    results[l]['jsd'].append(jsd)
                    results[l]['kld'].append(kld)

                    genres[genre]['jsd'].append(jsd)
        return results, genres

    def parse(self, references):
        result = []
        # keys = set(map(lambda x: (x['text-id'], x['paragraph-id']), references))
        keys = set(map(lambda x: x['text-id'], references))

        for key in keys:
            test = {'X':[], 'y':[], 'original':[], 'genre':'news'}

            # f = filter(lambda x: x['text-id'] == key[0] and x['paragraph-id'] == key[1], references)
            f = filter(lambda x: x['text-id'] == key, references)
            reference_ids = list(set(map(lambda x: x['reference-id'], f)))
            reference_ids.sort()
            for _id in reference_ids:
                g = filter(lambda x: x['reference-id'] == _id, f)

                X = {}
                for feature in self.features.keys():
                    X[feature] = g[0][feature]

                test['X'].append(numpy.array(self.reference2id[tuple(X.values())], dtype='int32'))
                test['original'].append(self.forms.index(g[0]['original-type']))
                test['genre'] = g[0]['genre']

                Y = []
                for y in self.forms:
                    Y.append(float(len(filter(lambda x: x['type'] == y, g))) / len(g))
                test['y'].append(numpy.array(Y, dtype='float32'))
            result.append(test)
        return result

    def contextwin(self, l, win):
        '''
        win :: int corresponding to the size of the window
        given a list of indexes composing a sentence

        l :: array containing the word indexes

        it will return a list of list of indexes corresponding
        to context windows surrounding each word in the sentence
        '''
        assert (win % 2) == 1
        assert win >= 1
        l = list(l)

        lpadded = win // 2 * [-1] + l + win // 2 * [-1]
        out = [lpadded[i:(i + win)] for i in range(len(l))]

        assert len(out) == len(l)
        return out

    def minibatch(self, l, bs):
        '''
        l :: list of word idxs
        return a list of minibatches of indexes
        which size is equal to bs
        border cases are treated as follow:
        eg: [0,1,2,3] and bs = 3
        will output:
        [[0],[0,1],[0,1,2],[1,2,3]]
        '''
        out  = [l[:i] for i in xrange(1, min(bs,len(l)+1) )]
        out += [l[i-bs:i] for i in xrange(bs,len(l)+1) ]
        assert len(l) == len(out)
        return out


if __name__ == '__main__':
    features = {'syncat': utils.syntax2id.keys(),\
    'categorical-recency': utils.recency2id.keys()}
    # 'distractor': utils.distractor2id.keys(),\
    # 'previous': utils.distractor2id.keys(), \
    # 'givenness': utils.givenness2id.keys(), \
    # 'paragraph-givenness': utils.givenness2id.keys(),\
    # 'sentence-givenness': utils.givenness2id.keys()}
    # 'genre': utils.genres2id.keys()}
    # forms = ['name', 'pronoun', 'description', 'demonstrative', 'empty']
    forms = ['demonstrative', 'description', 'empty', 'name', 'pronoun']
    s = {'lr':0.1,
         'win':3, # number of words in the context window
         'bs':10, # number of backprop through time steps
         'nhidden':50, # number of hidden units
         'emb':50, # dimension of word embedding
         'nepochs':15}

    rnn = RNNScript(features, forms)
    results, genres = rnn.run(s['nhidden'], s['emb'], s['win'], s['bs'], s['lr'], s['nepochs'])

    entropy_actual, entropy_predicted = [], []
    major_actual, major_predicted, original = [], [], []
    rank_actual, rank_predicted = [], []
    jsds, klds, bcoeffs = [], [], []

    for genre in genres:
        print '\n'
        print genre
        print 20 * '-'
        print 'Major Accuracy: ', accuracy_score(genres[genre]['major_actual'], genres[genre]['major_predicted'])
        print 'Original Accuracy: ', accuracy_score(genres[genre]['original'], genres[genre]['major_predicted'])
        print 'JSD: ', numpy.mean(genres[genre]['jsd']), numpy.std(genres[genre]['jsd'])

    for l in results:
        # print '\n'
        # print l.upper()
        # print 20 * '-'
        # print 'Major Accuracy: ', accuracy_score(results[l]['major_actual'], results[l]['major_predicted'])
        # print 'Binary Accuracy: ', accuracy_score(results[l]['binary_actual'], results[l]['binary_predicted'])
        # print 'Bcoeff: ', np.mean(results[l]['bcoeff'])

        original.extend(results[l]['original'])
        major_actual.extend(results[l]['major_actual'])
        major_predicted.extend(results[l]['major_predicted'])

        rank_actual.extend(results[l]['rank_actual'])
        rank_predicted.extend(results[l]['rank_predicted'])

        entropy_actual.extend(results[l]['entropy_actual'])
        entropy_predicted.extend(results[l]['entropy_predicted'])

        bcoeffs.extend(results[l]['bcoeff'])
        jsds.extend(results[l]['jsd'])
        klds.extend(results[l]['kld'])

    print '\n'
    print 'GENERAL:'
    print 20 * '-'
    print 'Major Accuracy: ', accuracy_score(major_actual, major_predicted)
    print 'Original Accuracy: ', accuracy_score(original, major_predicted)
    print 'Bcoeff: ', numpy.mean(bcoeffs)
    print 'JSD: ', numpy.mean(jsds), numpy.std(jsds)
    print 'KLD: ', numpy.mean(klds)
    print 20 * '-'
    rank = []
    for actual, predicted in zip(rank_actual, rank_predicted):
        aux = spearmanr(actual, predicted)[0]
        if str(aux) == 'nan':
            rank.append(0)
        else:
            rank.append(aux)
    print 'Rank: ', numpy.mean(rank)

    print '\n'
    print spearmanr(entropy_actual, entropy_predicted)

    print '\n'
    print classification_report(major_actual, major_predicted)