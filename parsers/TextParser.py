# -*- coding: utf-8 -*-
__author__ = 'thiagocastroferreira'

from Parser import Parser

import xml.etree.ElementTree as ET

class TextParser(Parser):
    def __init__(self):
        super(Parser)
        self.references = {1:('Anschutz', 'PERSON'), 2:('Cunningham', 'PERSON'), 3:('Google', 'ORGANIZATION'),\
                           5:('Ticktock', 'BOOK'), 6:('The Cat in the Hat', 'MOVIE'), 7:('The 5.8 GHz 4-Line Phone System', 'PHONE'),\
                           8:('Cerro Aconcagua', 'LOCATION'), 9:('Amazon River','LOCATION'), 10:('Brazil', 'LOCATION'), \
                           11:('Clooney','PERSON'), 12:('Summers','PERSON'), 13:('Vladimir','PERSON'), \
                           14:('The Eyes of the Dragon', 'BOOK'), 15:('The Haunted Mansion', 'MOVIE'), \
                           16:('The 5.8 GHz 4-Line Phone System', 'PHONE'), \
                           17:('Cho Oyu Mountain', 'LOCATION'), 18:('Yellow River', 'LOCATION'),19:('Ethiopia', 'LOCATION'),\
                           21:('Dunst', 'PERSON'), 22:('Jeanine','PERSON'), 23:('SunCom Wireless Holdings, Inc.', 'ORGANIZATION'),\
                           24:('Airframe Book', 'BOOK'), 25:('Gothika Movie', 'MOVIE'), \
                           26:('The 5.8 GHz 4-Line Phone System', 'PHONE'), \
                           27:('Kangchenjunga', 'LOCATION'), 28:('Sao Francisco River', 'LOCATION'), 29:('Indonesia', 'LOCATION'), \
                           30:('Samuel', 'PERSON'), 31:('Mary', 'PERSON'), 32:('Hamas', 'ORGANIZATION'), \
                           33:('The Eyes of the Dragon', 'BOOK'), 34:('The Haunted Mansion', 'MOVIE'), \
                           35:('The 5.8 GHz 4-Line Phone System', 'PHONE'), \
                           36:('Makalu Mountain', 'LOCATION'), 37:('Nile River', 'LOCATION'), 38:('India', 'LOCATION')}

    def __call__(self):
        self.run()

    def run(self):
        lists = self.list_files(root = '../data/xmls')

        texts = []

        keys = lists.keys()
        keys.sort()
        for l in keys:
            for f in lists[l]:
                texts.append(self.parse_text(f, l))
        return texts

    def parse_text(self, xml, list_id):
        root = ET.parse(xml)
        root = root.getroot()
        id = root.attrib['ID']

        refex = self.references[int(id)]
        paragraphs = root.findall('PARAGRAPH')

        text = {}
        originals = []
        for p_i, p in enumerate(paragraphs):
            sentences = p.findall('SENTENCE')
            paragraph = ''
            for s_i, s in enumerate(sentences):
                for i, child in enumerate(s):
                    if child.tag == 'REFERENCE':
                        original = child.find('ORIGINAL-REFEX').find('REFEX').text
                        originals.append({'id': child.attrib['ID'], 'paragraph_id':p_i+1, 'sentence_id':s_i+1, \
                                          'reference': original, 'position':child.attrib['PARAGRAPH-POSITION'],\
                                          'entropy': child.attrib['ENTROPY']})
                        if child.attrib['SYNCAT'] == 'subj-det':
                            paragraph = paragraph + 'Thiago Castro' + '\'s '
                        else:
                            paragraph = paragraph + 'Thiago Castro' + ' '
                    elif child.tag == 'STRING':
                        paragraph = paragraph + child.text.strip() + ' '
            text[p_i+1] = paragraph
        return (list_id, id, text, originals, refex[1])