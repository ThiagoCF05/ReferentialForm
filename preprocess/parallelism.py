__author__ = 'thiagocastroferreira'

from parsers.ReferenceParser import ReferenceParser
import numpy as np
import json

if __name__ == '__main__':
    parser = ReferenceParser(append_other=False, append_refex=False)
    references = parser.run()
    del parser

    print 'Parallelism:'
    f = filter(lambda x: x['parallelism'] == 'false', references)
    print 'Non-parallel: ', len(f), np.mean(map(lambda x: x['entropy'], f))
    f = filter(lambda x: x['parallelism'] == 'true', references)
    print 'Parallel: ', len(f), np.mean(map(lambda x: x['entropy'], f))
    print 20 * '-'

    references = map(lambda x: {'context_id':unicode(x['text-id']), \
                                'slot_id':unicode(x['reference-id']), \
                                'parallelism':x['parallelism']}, references)

    json.dump(references, open('parallelism.json', 'w'), indent=4, separators=(',', ': '))