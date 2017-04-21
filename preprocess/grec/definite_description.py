__author__ = 'thiagocastroferreira'

from stanford_corenlp_pywrapper import CoreNLP
from parsers.grec.Parser import GRECParser
from nltk.tokenize import sent_tokenize

import os
import json

class DefineDescription(object):
    def __init__(self):
        self.proc = CoreNLP("coref")
        self.parser = GRECParser()

    def run(self, xml):
        text = self.parser.parse_plain_text(xml)

        assert text[3] == 'PERSON'

        # Get the first sentence of the wikipedia page
        first_sentence = sent_tokenize(text[1][1])[0]
        # Parse the first sentence
        parsed_sentence = self.proc.parse_doc(first_sentence)
        # Get the position of the verb to be
        verb_position = self.__verb_position__(parsed_sentence['sentences'][0]['lemmas'])
        # Get the closest noun phrase to the verb to be
        description, postag = self.__get_description__(parsed_sentence, verb_position)
        # Make the description shorter
        description = self.__trim__(description, postag)
        # Insert an definite determiner
        description = self.__determiner__(description, postag)
        return description

    def __determiner__(self, description, postag):
        try:
            if postag[0] != 'DT':
                description.insert(0, 'the')
            else:
                description[0] = 'the'
            return description
        except:
            return description

    def __trim__(self, description, postag):
        _description = []
        for w_i, word in enumerate(description):
            _description.append(word)
            if postag[w_i] == 'NN':
                break
        return _description

    def __get_description__(self, parsed_sentence, verb_position):
        # Definite description, its distance to the verb to be and its part of speech information
        description, distance, postag = [], 1000, []
        for entity in parsed_sentence['entities']:
            for mention in entity['mentions']:
                tokens = parsed_sentence['sentences'][0]['tokens']
                pos = parsed_sentence['sentences'][0]['pos']
                begin, end = mention['tokspan_in_sentence'][0], mention['tokspan_in_sentence'][1]

                if begin-verb_position >= 0:
                    if begin-verb_position < distance:
                        description = tokens[begin:end]
                        postag = pos[begin:end]
                        distance = begin-verb_position
                    elif begin-verb_position == distance:
                        if len(tokens[begin:end]) < len(description):
                            description = tokens[begin:end]
                            postag = pos[begin:end]
                            distance = begin-verb_position
        return description, postag

    def __verb_position__(self, lemmas):
        verb_position = 0
        for word in lemmas:
            if word == 'be':
                break
            verb_position += 1
        return verb_position

if __name__ == '__main__':
    root = '../../data/grec/oficial'
    texts = os.listdir(root)

    parser = DefineDescription()

    _json = []
    for text in filter(lambda x: x != '.DS_Store', texts):
        try:
            description = {'text':text, 'description':' '.join(parser.run(text))}
            _json.append(description)
        except:
            pass
    json.dump(_json, open('descriptions.json', 'w'), indent=4, separators=(',', ': '))


