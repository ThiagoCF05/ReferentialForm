__author__ = 'thiagocastroferreira'

from parsers.ReferenceParser import ReferenceParser
from parsers.grec.Parser import GRECParser
from sklearn.metrics import accuracy_score
from models.elman import ElmanRNN
from scipy.stats import entropy, spearmanr, rankdata

import itertools
import numpy
import time
import measure
import utils
import sys
import copy

class RNNGRECScript(object):
    def __init__(self, features, forms):
        self.features = features
        self.forms = forms
        self.n = len(self.forms)

        self.reference2id = list(set(list(itertools.product(*self.features.values()))))
        self.reference2id = [dict([(self.features.keys()[i], e) for i, e in enumerate(combination)]) for combination in self.reference2id]
        self.reference2id = dict([(tuple(reference.values()), id) for id, reference in enumerate(self.reference2id)])

        parser = GRECParser('../../data/grec/oficial')
        trainset = parser.run()
        # trainset = parser.parse_ner(trainset)

        parser = ReferenceParser(False, True, '../../data/xmls')
        testset = parser.run()
        del parser

        self.train = self.parse_train(trainset)
        self.test = self.parse(testset)

    def run(self, nh, de, cs, bs, lr, epochs):
        genres = {'news':{'major_actual':[], 'major_predicted': [], 'original': [], 'jsd':[]}, \
                  'review':{'major_actual':[], 'major_predicted': [], 'original': [], 'jsd': []}, \
                  'wiki':{'major_actual':[], 'major_predicted': [], 'original': [], 'jsd':[]}}
        results = {'entropy_actual':[], 'entropy_predicted':[], \
                      'major_actual':[], 'major_predicted': [], 'original':[], \
                      'binary_actual':[], 'binary_predicted':[], \
                      'rank_actual':[], 'rank_predicted':[], \
                      'bcoeff': [], 'jsd':[], 'kld':[]}

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
                       for x in self.test]

        for i, x in enumerate(self.test):
            major_actual = numpy.argmax(x['y'], axis=1)
            major_predicted = numpy.argmax(predictions[i], axis=1)

            # print major_actual, major_predicted
            results['original'].extend(x['original'])
            results['major_actual'].extend(major_actual)
            results['major_predicted'].extend(major_predicted)

            genre = x['genre']
            genres[genre]['original'].extend(x['original'])
            genres[genre]['major_actual'].extend(major_actual)
            genres[genre]['major_predicted'].extend(major_predicted)

            for j, gap in enumerate(x['y']):
                bcoeff = round(measure.bhattacharya(numpy.array(x['y'][j]), numpy.array(predictions[i][j])), 2)
                jsd = measure.JSD(numpy.array(x['y'][j]), numpy.array(predictions[i][j]), 2)
                kld = round(entropy(numpy.array(x['y'][j]), numpy.array(predictions[i][j]), 2), 2)

                results['entropy_actual'].append(entropy(numpy.array(x['y'][j])))
                results['entropy_predicted'].append(entropy(numpy.array(predictions[i][j])))

                results['rank_actual'].append(rankdata(x['y'][j]))
                results['rank_predicted'].append(rankdata(predictions[i][j]))

                results['bcoeff'].append(bcoeff)
                results['jsd'].append(jsd)
                results['kld'].append(kld)

                genres[genre]['jsd'].append(jsd)
        return results, genres

    # parse train with a global distribution over the referential forms
    def parse_train(self, references):
        result = []
        keys = set(map(lambda x: x['text-id'], references))

        for key in keys:
            test = {'X':[], 'y':[]}

            f = filter(lambda x: x['text-id'] == key, references)
            reference_ids = list(set(map(lambda x: x['reference-id'], f)))
            reference_ids.sort()
            for _id in reference_ids:
                g = filter(lambda x: x['reference-id'] == _id, f)

                X = {}
                z = copy.copy(references)
                for feature in self.features.keys():
                    X[feature] = g[0][feature]
                    z = filter(lambda x: x[feature] == X[feature], z)

                test['X'].append(numpy.array(self.reference2id[tuple(X.values())], dtype='int32'))

                Y = []
                for y in self.forms:
                    Y.append(float(len(filter(lambda x: x['type'] == y, z))) / len(z))
                test['y'].append(numpy.array(Y, dtype='float32'))
            result.append(test)
        return result

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
    features = {'syncat': utils.syntax2id.keys(), \
                # 'categorical-recency': utils.recency2id.keys(), \
                'givenness': utils.givenness2id.keys(), \
                'paragraph-givenness': utils.givenness2id.keys(), \
                'sentence-givenness': utils.givenness2id.keys()}
    # forms = ['name', 'pronoun', 'description', 'demonstrative', 'empty']
    forms = ['demonstrative', 'description', 'empty', 'name', 'pronoun']
    s = {'lr':0.1,
         'win':3, # number of words in the context window
         'bs':10, # number of backprop through time steps
         'nhidden':50, # number of hidden units
         'emb':50, # dimension of word embedding
         'nepochs':15}

    rnn = RNNGRECScript(features, forms)
    results, genres = rnn.run(s['nhidden'], s['emb'], s['win'], s['bs'], s['lr'], s['nepochs'])

    for genre in genres:
        print '\n'
        print genre
        print 20 * '-'
        print 'Major Accuracy: ', accuracy_score(genres[genre]['major_actual'], genres[genre]['major_predicted'])
        print 'Original Accuracy: ', accuracy_score(genres[genre]['original'], genres[genre]['major_predicted'])
        print 'JSD: ', numpy.mean(genres[genre]['jsd']), numpy.std(genres[genre]['jsd'])

    print '\n'
    print 'GENERAL:'
    print 20 * '-'
    print 'Major Accuracy: ', accuracy_score(results['major_actual'], results['major_predicted'])
    print 'Original Accuracy: ', accuracy_score(results['original'], results['major_predicted'])
    print 'Bcoeff: ', numpy.mean(results['bcoeff'])
    print 'JSD: ', numpy.mean(results['jsd']), numpy.std(results['jsd'])
    print 'KLD: ', numpy.mean(results['kld'])
    print 20 * '-'
    rank = []
    for actual, predicted in zip(results['rank_actual'], results['rank_predicted']):
        aux = spearmanr(actual, predicted)[0]
        if str(aux) == 'nan':
            rank.append(0)
        else:
            rank.append(aux)
    print 'Rank: ', numpy.mean(rank)

    print '\n'
    print spearmanr(results['entropy_actual'], results['entropy_predicted'])

    # print '\n'
    # print classification_report(results['major_actual'], results['major_predicted'])