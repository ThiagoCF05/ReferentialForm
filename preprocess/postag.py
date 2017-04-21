__author__ = 'thiagocastroferreira'

from parsers.ReferenceParser import ReferenceParser
import numpy as np

if __name__ == '__main__':
    parser = ReferenceParser(append_other=False, append_refex=False)
    references = parser.run()
    del parser

    postags = set(map(lambda x: x['pos-bigram'], references))

    f = map(lambda x: x['entropy'], filter(lambda x: x['pos-bigram'] in ['NN', 'NNS'], references))
    print 'NN', round(np.mean(f), 4), round(np.std(f), 4), len(f)

    f = map(lambda x: x['entropy'], filter(lambda x: x['pos-bigram'] in ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ'], references))
    print 'V', round(np.mean(f), 4), round(np.std(f), 4), len(f)

    f = map(lambda x: x['entropy'], filter(lambda x: x['pos-bigram'] in ['\'\'', ',', ':', '.'], references))
    print 'PUNC', round(np.mean(f), 4), round(np.std(f), 4), len(f)

    f = map(lambda x: x['entropy'], filter(lambda x: x['pos-bigram'] in ['JJ', 'JJR', 'JJS'], references))
    print 'JJ', round(np.mean(f), 4), round(np.std(f), 4), len(f)

    f = map(lambda x: x['entropy'], filter(lambda x: x['pos-bigram'] == 'IN', references))
    print 'IN', round(np.mean(f), 4), round(np.std(f), 4), len(f)

    f = map(lambda x: x['entropy'], filter(lambda x: x['pos-bigram'] == 'DT', references))
    print 'DT', round(np.mean(f), 4), round(np.std(f), 4), len(f)

    f = map(lambda x: x['entropy'], filter(lambda x: x['pos-bigram'] == 'TO', references))
    print 'TO', round(np.mean(f), 4), round(np.std(f), 4), len(f)

    f = map(lambda x: x['entropy'], filter(lambda x: x['pos-bigram'] in ['RB', 'WRB'], references))
    print 'RB', round(np.mean(f), 4), round(np.std(f), 4), len(f)

    f = map(lambda x: x['entropy'], filter(lambda x: x['pos-bigram'] == 'CC', references))
    print 'CC', round(np.mean(f), 4), round(np.std(f), 4), len(f)

    f = map(lambda x: x['entropy'], filter(lambda x: x['pos-bigram'] == '-NONE-', references))
    print 'None', round(np.mean(f), 4), round(np.std(f), 4), len(f)

    f = map(lambda x: x['entropy'], filter(lambda x: x['genre'] == 'news', references))
    print f
    f = map(lambda x: x['entropy'], filter(lambda x: x['genre'] == 'review', references))
    print f
    f = map(lambda x: x['entropy'], filter(lambda x: x['genre'] == 'wiki', references))
    print f