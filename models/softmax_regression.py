__author__ = 'thiagocastroferreira'

import theano
import theano.tensor as T
from collections import OrderedDict

import numpy as np

class SoftmaxRegression(object):
    def __init__(self, ni, nh):
        self.W = theano.shared(np.random.uniform(-0.1, 0.1, (ni, nh)).astype(theano.config.floatX))
        self.b = theano.shared(np.zeros(nh, dtype=theano.config.floatX))
        self.params = [self.W, self.b]

        X = T.dmatrix('X')
        y = T.dmatrix('y')

        soft = T.nnet.softmax(T.dot(X, self.W) + self.b)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        l2 = T.sum(self.W ** 2)
        nll = T.mean(T.nnet.categorical_crossentropy(soft, y)) + l2
        gradients = T.grad(nll, self.params)
        updates = OrderedDict((p, p-lr*g) for p, g in zip(self.params, gradients))

        # theano functions
        self.classify = theano.function(inputs=[X], outputs=soft)

        self.train = theano.function(inputs=[X, y, lr], outputs=nll, updates=updates)

        self.validate = theano.function(inputs=[X, y], outputs=nll)

class SoftmaxMLP(object):
    def __init__(self, ni, nh, ns):
        self.Wh = theano.shared(np.random.uniform(-0.01, 0.01, (ni, nh)).astype(theano.config.floatX))
        self.bh = theano.shared(np.zeros(nh, dtype=theano.config.floatX))

        self.W = theano.shared(np.random.uniform(-0.01, 0.01, (nh, ns)).astype(theano.config.floatX))
        self.b = theano.shared(np.zeros(ns, dtype=theano.config.floatX))
        self.params = [self.Wh, self.bh, self.W, self.b]

        X = T.dmatrix('X')
        y = T.dmatrix('y')

        h = T.nnet.sigmoid(T.dot(X, self.Wh) + self.bh)
        soft = T.nnet.softmax(T.dot(h, self.W) + self.b)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        l2 = T.sum(self.W ** 2)
        nll = T.mean(T.nnet.categorical_crossentropy(soft, y)) + l2
        gradients = T.grad(nll, self.params)
        updates = OrderedDict((p, p-lr*g) for p, g in zip(self.params, gradients))

        # theano functions
        self.classify = theano.function(inputs=[X], outputs=soft)

        self.train = theano.function(inputs=[X, y, lr], outputs=nll, updates=updates)

        self.validate = theano.function(inputs=[X, y], outputs=nll)