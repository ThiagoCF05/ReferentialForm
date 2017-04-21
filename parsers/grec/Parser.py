__author__ = 'thiagocastroferreira'

from collections import Counter

import xml.etree.ElementTree as ET
import utils
import nltk
import os
import json

class GRECParser():
    def __init__(self, root = '../../data/grec/oficial'):
        self.root = root

    def run(self):
        references = []

        for f in self.list_files():
            print f, '\r',
            references.extend(self.parse_text(f))

        return references

    def list_files(self):
        dirs = filter(lambda x: x != '.DS_Store', os.listdir(self.root))
        return dirs

    def run_plain_parse(self):
        texts = []

        for f in self.list_files():
            print f, '\r',
            texts.append(self.parse_plain_text(f))

        return texts

    def parse_plain_text(self, xml):
        root = ET.parse(os.path.join(self.root, xml))
        root = root.getroot()
        id = root.attrib['ID']

        paragraphs = root.findall('PARAGRAPH')

        text = {}
        originals = []
        nertype = 'LOCATION'

        for p_i, p in enumerate(paragraphs):
            sentences = p.findall('SENTENCE')
            paragraph = ''
            for s_i, s in enumerate(sentences):
                for i, child in enumerate(s):
                    if child.tag == 'REFERENCE':
                        if child.attrib['SEMCAT'] == 'person':
                            nertype = 'PERSON'

                        original = child.find('ORIGINAL-REFEX').find('REFEX').text
                        originals.append({'id': child.attrib['ID'], 'paragraph_id':p_i+1, 'sentence_id':s_i+1, \
                                          'reference': original, 'position':child.attrib['PARAGRAPH-POSITION']})
                        # if child.attrib['SYNCAT'] == 'subj-det':
                        #     paragraph = paragraph + 'Thiago Castro' + '\'s '
                        # else:
                        #     paragraph = paragraph + 'Thiago Castro' + ' '
                        paragraph = paragraph + original + ' '
                    elif child.tag == 'STRING':
                        paragraph = paragraph + child.text.strip() + ' '
            text[p_i+1] = paragraph
        return (id, text, originals, nertype)

    def parse_frequencies(self, references):
        texts = set(map(lambda x: x['text-id'], references))
        for text in texts:
            text_references = filter(lambda x: x['text-id'] == text, references)
            for reference in text_references:
                previous = filter(lambda x: int(x['reference-id']) < int(reference['reference-id']), text_references)
                previous = map(lambda x: x['type'], previous)

                reference['freq-name'] = Counter(previous)['name']
                reference['freq-pronoun'] = Counter(previous)['pronoun']
                reference['freq-description'] = Counter(previous)['description']
                reference['freq-demonstrative'] = 0
                reference['freq-empty'] = Counter(previous)['empty']
        return references

    def parse_ner(self, references):
        ner = json.load(open('../../data/grec/grec-ner.json'))

        texts = set(map(lambda x: x['text-id'], references))
        for text in texts:
            f = filter(lambda x: int(x['textid']) == int(text), ner)
            text_references = filter(lambda x: x['text-id'] == text, references)
            for reference in text_references:
                g = filter(lambda x: int(x['id']) == int(reference['reference-id']), f)
                if len(g) == 0:
                    # print reference['text-id'], reference['reference-id']
                    reference['num-entities'] = 0
                    reference['num-entities-same'] = 0
                    reference['dist-entities'] = 0
                    reference['dist-entities-same'] = 0

                    reference['previous'] = 'false'
                    reference['distractor'] = 'false'
                else:
                    reference['num-entities'] = g[0]['entities']
                    reference['num-entities-same'] = g[0]['entities_same_type']
                    reference['dist-entities'] = g[0]['distance']
                    reference['dist-entities-same'] = g[0]['distance_same_type']

                    if reference['num-entities'] > 0:
                        reference['previous'] = 'true'
                    else:
                        reference['previous'] = 'false'

                    if reference['num-entities-same'] > 0:
                        reference['distractor'] = 'true'
                    else:
                        reference['distractor'] = 'false'
        return references

    def parse_text(self, xml):
        references = []
        root = ET.parse(os.path.join(self.root, xml))
        root = root.getroot()
        paragraphs = root.findall('PARAGRAPH')

        for p in paragraphs:
            last_string = '*'
            last_syntax = '*'
            sentences = p.findall('SENTENCE')
            for s in sentences:
                for i, child in enumerate(s):
                    if child.tag == 'STRING':
                        last_string = child.text
                    elif child.tag == 'REFERENCE':
                        ref = child
                        reference = {}
                        reference['paragraph-id'] = int(p.attrib['ID'])
                        reference['text-id'] = root.attrib['ID']

                        reference['syncat'] = ref.attrib['SYNCAT']
                        reference['-syncat'] = last_syntax
                        last_syntax = reference['syncat']

                        reference['paragraph-recency'] = int(ref.attrib['PARAGRAPH-RECENCY'])
                        reference['categorical-recency'] = utils.recency(int(ref.attrib['PARAGRAPH-RECENCY']))
                        reference['paragraph-position'] = int(ref.attrib['PARAGRAPH-POSITION'])
                        reference['sentence-recency'] = int(ref.attrib['SENTENCE-RECENCY'])
                        reference['sentence-position'] = int(ref.attrib['SENTENCE-POSITION'])
                        reference['reference-id'] = int(ref.attrib['ID'])

                        original = ref.find('ORIGINAL-REFEX').find('REFEX')
                        reference['type'] = original.attrib['TYPE']
                        reference['head'] = original.attrib['HEAD']
                        reference['case'] = original.attrib['CASE']
                        reference['emphatic'] = original.attrib['EMPHATIC']
                        reference['refex'] = original.text

                        reference['name'] = xml

                        if ref.attrib['SEMCAT'] == 'person':
                            reference['animacy'] = 'animate'
                        else:
                            reference['animacy'] = 'inanimate'

                        if int(reference['reference-id']) == 1:
                            reference['givenness'] = 'new'
                        else:
                            reference['givenness'] = 'given'

                        if int(reference['paragraph-recency']) == 0:
                            reference['paragraph-givenness'] = 'new'
                        else:
                            reference['paragraph-givenness'] = 'given'

                        if int(reference['sentence-recency']) == 0:
                            reference['sentence-givenness'] = 'new'
                        else:
                            reference['sentence-givenness'] = 'given'

                        # text = nltk.word_tokenize(last_string)
                        # pos_tag = nltk.pos_tag(text)
                        #
                        # reference['clause'] = 'regular'
                        #
                        # if len(text) == 0:
                        #     reference['bigram'] = '*'
                        #     reference['pos-bigram'] = '-NONE-'
                        #
                        #     reference['trigram'] = '*'
                        #     reference['pos-trigram'] = '-NONE-'
                        # else:
                        #     reference['bigram'] = text[-1]
                        #     reference['pos-bigram'] = pos_tag[-1][1]
                        #
                        #     if reference['pos-bigram'] in ['CC']:
                        #         reference['clause'] = 'coordinate'
                        #     elif reference['pos-bigram'] in ['IN', 'WRB', 'WDT', 'WP', 'WP$']:
                        #         reference['clause'] = 'subordinate'
                        #
                        #     try:
                        #         reference['trigram'] = text[-2]
                        #         reference['pos-trigram'] = pos_tag[-2][1]
                        #     except:
                        #         reference['trigram'] = '*'
                        #         reference['pos-trigram'] = '-NONE-'
                        #
                        # if s[i+1].tag == 'STRING':
                        #     text = nltk.word_tokenize(s[i+1].text)
                        #     pos_tag = nltk.pos_tag(text)
                        #
                        #     if len(text) == 0:
                        #         reference['+bigram'] = '*'
                        #         reference['+pos-bigram'] = '-NONE-'
                        #
                        #         reference['+trigram'] = '*'
                        #         reference['+pos-trigram'] = '-NONE-'
                        #     else:
                        #         reference['+bigram'] = text[0]
                        #         reference['+pos-bigram'] = pos_tag[0][1]
                        #
                        #         try:
                        #             reference['+trigram'] = text[1]
                        #             reference['+pos-trigram'] = pos_tag[1][1]
                        #         except:
                        #             reference['+trigram'] = '*'
                        #             reference['+pos-trigram'] = '-NONE-'
                        # else:
                        #     reference['+bigram'] = '*'
                        #     reference['+pos-bigram'] = '-NONE-'
                        #
                        #     reference['+trigram'] = '*'
                        #     reference['+pos-trigram'] = '-NONE-'
                        references.append(reference)

        return references