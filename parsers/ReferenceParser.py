__author__ = 'thiagocastroferreira'

from collections import Counter

import xml.etree.ElementTree as ET
import copy
from Parser import Parser
import utils
import nltk
import json

class ReferenceParser(Parser):
    def __init__(self, append_other = False, append_refex = True, root = '../data/xmls'):
        super(Parser)
        self.append_other = append_other
        self.append_refex = append_refex
        self.root = root

    def run(self):
        lists = self.list_files(self.root)

        references = []

        keys = lists.keys()
        keys.sort()
        for l in keys:
            for f in lists[l]:
                references.extend(self.parse_text(f, l))

        return references

    def parse_frequencies(self, references):
        texts = set(map(lambda x: x['text-id'], references))
        for text in texts:
            text_references = filter(lambda x: x['text-id'] == text, references)
            for reference in text_references:
                # previous = filter(lambda x: int(x['reference-id']) < int(reference['reference-id']) \
                #                             and int(x['participant-id']) == int(reference['participant-id']), text_references)
                previous = filter(lambda x: int(x['reference-id']) < int(reference['reference-id']), text_references)
                previous = map(lambda x: x['type'], previous)

                reference['freq-name'] = Counter(previous)['name']
                reference['freq-pronoun'] = Counter(previous)['pronoun']
                reference['freq-description'] = Counter(previous)['description']
                reference['freq-demonstrative'] = Counter(previous)['demonstrative']
                reference['freq-empty'] = Counter(previous)['empty']
        return references

    def parse_ner(self, references):
        try:
            ner = json.load(open('../data/ner.json'))
        except:
            ner = json.load(open('../../data/ner.json'))

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


    def parse_last_mention(self, references):
        texts = set(map(lambda x: x['text-id'], references))
        for text in texts:
            text_references = filter(lambda x: x['text-id'] == text, references)
            for reference in text_references:
                previous = filter(lambda x: int(x['reference-id']) == int(reference['reference-id'])-1 , text_references)
                previous = map(lambda x: x['type'], previous)

                if len(previous) > 0:
                    reference['last-type'] = previous[0]
                else:
                    reference['last-type'] = 'other'
        return references

    def parse_text(self, xml, list_id):
        references = []
        root = ET.parse(xml)
        root = root.getroot()

        paragraphs = root.findall('PARAGRAPH')

        for p in paragraphs:
            last_string = '*'
            sentences = p.findall('SENTENCE')
            last_syntax = (1, '*')
            for s in sentences:
                for i, child in enumerate(s):
                    if child.tag == 'STRING':
                        last_string = child.text
                    elif child.tag == 'REFERENCE':
                        ref = child
                        reference = {}
                        reference['list-id'] = list_id
                        reference['paragraph-id'] = int(p.attrib['ID'])
                        reference['sentence-id'] = int(s.attrib['ID'].split('.')[1])
                        reference['text-id'] = root.attrib['ID']
                        reference['file'] = xml.split('/')[-1].split('.')[0]

                        reference['syncat'] = ref.attrib['SYNCAT']
                        reference['-syncat'] = last_syntax[1]

                        reference['parallelism'] = 'false'
                        if reference['sentence-id']-last_syntax[0] <= 1:
                            if reference['syncat'] == reference['-syncat']:
                                reference['parallelism'] = 'true'

                        last_syntax = (reference['sentence-id'], reference['syncat'])

                        reference['entropy'] = float(ref.attrib['ENTROPY'])
                        reference['paragraph-recency'] = int(ref.attrib['PARAGRAPH-RECENCY'])
                        reference['categorical-recency'] = utils.recency(int(ref.attrib['PARAGRAPH-RECENCY']))
                        reference['paragraph-position'] = int(ref.attrib['PARAGRAPH-POSITION'])
                        reference['sentence-recency'] = int(ref.attrib['SENTENCE-RECENCY'])
                        reference['sentence-position'] = int(ref.attrib['SENTENCE-POSITION'])
                        reference['reference-id'] = int(ref.attrib['ID'])
                        reference['original-type'] = ref.find('ORIGINAL-REFEX').find('REFEX').attrib['TYPE']
                        # reference['topic'] = ref.attrib['TOPIC']
                        # reference['competitor'] = ref.attrib['COMPETITOR']
                        reference['animacy'] = root.attrib['TOPIC-ANIMACY']

                        reference['genre'] = root.attrib['GENRE']

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

                        text = nltk.word_tokenize(last_string)
                        pos_tag = nltk.pos_tag(text)

                        reference['clause'] = 'regular'

                        if len(text) == 0:
                            reference['bigram'] = '*'
                            reference['pos-bigram'] = '-NONE-'

                            reference['trigram'] = '*'
                            reference['pos-trigram'] = '-NONE-'
                        else:
                            reference['bigram'] = text[-1]
                            reference['pos-bigram'] = pos_tag[-1][1]

                            if reference['pos-bigram'] in ['CC']:
                                reference['clause'] = 'coordinate'
                            elif reference['pos-bigram'] in ['IN', 'WRB', 'WDT', 'WP', 'WP$']:
                                reference['clause'] = 'subordinate'

                            try:
                                reference['trigram'] = text[-2]
                                reference['pos-trigram'] = pos_tag[-2][1]
                            except:
                                reference['trigram'] = '*'
                                reference['pos-trigram'] = '-NONE-'

                        if s[i+1].tag == 'STRING':
                            text = nltk.word_tokenize(s[i+1].text)
                            pos_tag = nltk.pos_tag(text)

                            if len(text) == 0:
                                reference['+bigram'] = '*'
                                reference['+pos-bigram'] = '-NONE-'

                                reference['+trigram'] = '*'
                                reference['+pos-trigram'] = '-NONE-'
                            else:
                                reference['+bigram'] = text[0]
                                reference['+pos-bigram'] = pos_tag[0][1]

                                try:
                                    reference['+trigram'] = text[1]
                                    reference['+pos-trigram'] = pos_tag[1][1]
                                except:
                                    reference['+trigram'] = '*'
                                    reference['+pos-trigram'] = '-NONE-'
                        else:
                            reference['+bigram'] = '*'
                            reference['+pos-bigram'] = '-NONE-'

                            reference['+trigram'] = '*'
                            reference['+pos-trigram'] = '-NONE-'
                        if self.append_refex:
                            for refex in ref.find('PARTICIPANTS-REFEX').findall('REFEX'):
                                row = copy.copy(reference)
                                row['participant-id'] = refex.attrib['PARTICIPANT-ID']
                                row['type'] = refex.attrib['TYPE']
                                row['refex'] = refex.text
                                if row['type'] == 'other':
                                    if self.append_other == True:
                                        references.append(row)
                                else:
                                    references.append(row)
                        else:
                            references.append(reference)
        return references