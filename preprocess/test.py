__author__ = 'thiagocastroferreira'

from parsers.ReferenceParser import ReferenceParser
from collections import Counter

if __name__ == '__main__':
    parser = ReferenceParser()
    references = parser.run()
    del parser

    num = len(filter(lambda x: x['original-type'] != x['type'], references))

    print float(num) / len(references)

    texts = set(map(lambda x: x['text-id'], references))
    num, dem = 0, 0
    for text in texts:
        f = filter(lambda x: x['text-id'] == text, references)
        gaps = set(map(lambda x: x['reference-id'], f))

        for gap in gaps:
            dem += 1
            g = filter(lambda x: x['reference-id'] == gap, f)

            c = Counter(map(lambda x: x['type'], g))
            major = filter(lambda x: c[x] == max(c.values()), c.keys())[0]

            if major != g[0]['original-type']:
                num += 1

    print float(num) / dem