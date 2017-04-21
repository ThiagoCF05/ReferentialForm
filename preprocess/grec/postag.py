__author__ = 'thiagocastroferreira'

from parsers.grec.Parser import GRECParser
import numpy as np
import measure

if __name__ == '__main__':
    references = GRECParser().run()

    postags = set(map(lambda x: x['pos-bigram'], references))

    for postag in postags:
        f = map(lambda x: x['type'], filter(lambda x: x['pos-bigram'] == postag, references))
        print postag, round(measure.nentropy(f, ['name', 'pronoun', 'description', 'demonstrative', 'empty']), 4), len(f)