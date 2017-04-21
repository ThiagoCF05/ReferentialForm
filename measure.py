__author__ = 'thiagocastroferreira'

import numpy as np
import scipy.stats

''' Divergence measures '''
def xlog(xi, yi):
    if xi == 0 or yi == 0:
        return 0
    else:
        return xi*np.log(float(xi)/float(yi))

def KLD(x,y):
    """ Kullback Leibler divergence """
    return sum([xlog(xi, yi) for xi, yi in zip(x, y)])

def JSD(p, q, base):
    """ Jensen Shannon divergence """
    p = np.array(p)
    q = np.array(q)
    return np.sqrt(0.5* scipy.stats.entropy(p, 0.5*(p + q), base) + 0.5 * scipy.stats.entropy(q, 0.5*(p + q), base))

def bhattacharya(p, q):
    """ Bhattacharya coefficient """
    p = np.array(p)
    q = np.array(q)
    return np.sum(np.sqrt(p) * np.sqrt(q))

def crossentropy(p, q):
    entropy = 0.0
    for pi, qi in zip(p, q):
        if qi != 0:
            entropy = entropy + (pi * np.log(qi))
    return -entropy

''' Information Gain functions'''
''' Entropy '''
def entropy(sequence, events):
    distribution = []
    for event in events:
        count = len(filter(lambda x: x == event, sequence))
        distribution.append(float(count) / len(sequence))
    return scipy.stats.entropy(distribution)

''' Normalized Entropy '''
def nentropy(sequence, events):
    distribution = []
    for event in events:
        count = len(filter(lambda x: x == event, sequence))
        distribution.append(float(count) / len(sequence))

    entropy = 0.0
    for event in distribution:
        if event != 0.0:
            entropy = entropy + ((event * np.log(event)) / np.log(len(events)))
    return -entropy

''' Information Gain '''
def gain(references, features, classes):
    gains = {}
    S = entropy(map(lambda x: x['type'], references), classes)

    for feature in features:
        gains[feature] = S
        for v in features[feature]:
            S_v = filter(lambda x: x[feature] == v, references)
            if len(S_v) != 0:
                gains[feature] -= (float(len(S_v)) / len(references)) * entropy(map(lambda x: x['type'], S_v), classes)
    return gains

''' Ranking '''
def rank(events):
    events = dict(map(lambda x: (x, round(events[x], 2)), events.keys()))
    ranking = dict(map(lambda x: (x, 0), events.keys()))

    pos = 1
    while len(events) > 0:
        maximum = max(events.values())

        filtered = filter(lambda x: events[x] == maximum, events.keys())
        for f in filtered:
            ranking[f] = pos
            del events[f]
        pos = pos + 1
    return ranking