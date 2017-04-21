# Create the json coreference representation

__author__ = 'thiagocastroferreira'

from parsers.TextParser import TextParser

from stanford_corenlp_pywrapper import CoreNLP

import json
import os

proc = CoreNLP("coref")

texts = TextParser().run()

for text in texts:
    print text
    file = os.path.join('../data/coreferences', str(text[0]) + '_' + str(text[1]) + '.txt')
    file_json = os.path.join('../data/coreferences-json', str(text[0]) + '_' + str(text[1]) + '.json')
    f = open(file, 'w')

    parsed = proc.parse_doc(text[2])
    json.dump(parsed, open(file_json, 'w'))

    keys = filter(lambda x: x not in ['tokens', 'pos'], parsed['sentences'][0].keys())
    for sentence in parsed['sentences']:
        for key in sentence.keys():
            if key not in ['tokens', 'pos']: del sentence[key]

    references = []
    for entity in parsed['entities']:
        f.write('ENTITY: ' + str(entity['entityid']))
        f.write('\n')
        f.write(20 * '-')
        f.write('\n')
        for mention in entity['mentions']:
            begin = int(mention['tokspan_in_sentence'][0])
            end = int(mention['tokspan_in_sentence'][1])
            mention['reference'] = parsed['sentences'][mention['sentence']]['tokens'][begin:end]

            try:
                f.write((' ').join(mention['reference']))
            except:
                aux = map(lambda x: x.encode('utf-8'), mention['reference'])
                f.write((' ').join(aux))
                del aux
            f.write('\n')
            f.write(str(mention['sentence']))
            f.write('\n')
            f.write(str(mention['mentiontype']))
            f.write('\n')
            f.write(mention['gender'])
            f.write('\n')
            f.write(mention['number'])
            f.write('\n')
            f.write(mention['animacy'])
            f.write('\n')
            f.write(20 * '-')
            f.write('\n')
        f.write('\n')

    f.close()