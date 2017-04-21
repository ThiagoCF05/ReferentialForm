__author__ = 'thiagocastroferreira'

from parsers.TextParser import TextParser
from stanford_corenlp_pywrapper import CoreNLP
import json

class DistractorScript(object):
    def __init__(self):
        self.texts = TextParser().run()
        self.proc = CoreNLP("ner")

    def run(self):
        results = []
        for text in self.texts:
            results.extend(self.parse_text(text))
        return results

    def parse_text(self, text):
        listid, textid, paragraphs, references, nertype = text

        results = []
        for paragraph in paragraphs:
            f = filter(lambda x: x['paragraph_id'] == paragraph, references)
            if len(f) > 0:
                results.extend(self.parse_paragraph(textid, paragraphs[paragraph], f, nertype))
        return results

    def caldistances(self, text, _index, _same_type_index):
        dist = text[_index[0]:_index[1]].strip().replace('.', '').replace('!', '').replace('?', '')\
            .replace(',', '').replace(',', '').replace(';', '').replace(':', '')
        dist = len(dist.split())

        same_type_dist = text[_same_type_index[0]:_same_type_index[1]].strip().replace('.', '')\
            .replace('!', '').replace('?', '').replace(',', '').replace(',', '').replace(';', '').replace(':', '')
        same_type_dist = len(same_type_dist.split())
        return dist, same_type_dist

    def parse_paragraph(self, textid, paragraph, references, nertype):
        r = self.proc.parse_doc(paragraph)
        types = []
        pos, dist, sdist = 0, -1, -1
        results = []
        for s_i, sentence in enumerate(r['sentences']):
            for mention in sentence['entitymentions']:
                begin, end = mention['charspan'][0], mention['charspan'][1]
                entity = paragraph[begin:end]

                if 'Thiago Castro' in entity.strip():
                    entities = len(types)
                    entities_same_type = len(filter(lambda x: x == nertype, types))
                    distance, sdistance = self.caldistances(paragraph, (dist, begin), (sdist, begin))

                    result = {'id':references[pos]['id'],'textid':textid, \
                              'entities':entities,'entities_same_type':entities_same_type, \
                              'distance':distance, 'distance_same_type':sdistance, \
                              'original':references[pos]['reference'], 'entropy':float(references[pos]['entropy'])}
                    results.append(result)
                    types = []
                    dist, sdist = -1, -1
                    pos += 1
                else:
                    types.append(mention['type'])
                    dist = end
                    if mention['type'] == nertype:
                        sdist = end

        return results

results = DistractorScript().run()
for result in results:
    print result
json.dump(results, open('ner.json', 'w'))