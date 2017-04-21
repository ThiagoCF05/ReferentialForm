__author__ = 'thiagocastroferreira'

from parsers.ReferenceParser import ReferenceParser
import numpy as np

if __name__ == '__main__':
    parser = ReferenceParser(append_other=False, append_refex=False)
    references = parser.run()
    del parser

    for reference in references:
        if reference['paragraph-givenness'] == 'new':
            reference['focus'] = reference['topic']
        else:
            old = filter(lambda x: x['text-id'] == reference['text-id'] and \
                                   int(x['reference-id'] == int(reference['reference-id']-1)), references)[0]
            reference['focus'] = old['topic']


    print 'Topicality:'
    print 'Non-topic: ', np.mean(map(lambda x: x['entropy'], filter(lambda x: x['topic'] == 'false', references)))
    print 'Topic: ', np.mean(map(lambda x: x['entropy'], filter(lambda x: x['topic'] == 'true', references)))
    print 20 * '-'
    print 'Focus:'
    print 'Non-focus: ', np.mean(map(lambda x: x['entropy'], filter(lambda x: x['focus'] == 'false', references)))
    print 'Focus: ', np.mean(map(lambda x: x['entropy'], filter(lambda x: x['focus'] == 'true', references)))