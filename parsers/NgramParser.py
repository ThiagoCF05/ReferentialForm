__author__ = 'thiagocastroferreira'

from Parser import Parser

import xml.etree.ElementTree as ET
import nltk

class NgramParser(Parser):
    def __init__(self):
        super(Parser)

    def __call__(self):
        self.run()

    def run(self):
        lists = self.list_files()

        texts = []

        keys = lists.keys()
        keys.sort()
        for l in keys:
            for f in lists[l]:
                texts.extend(self.parse_text(f))

        ngram = nltk.bigrams(texts)
        return [nltk.FreqDist(texts), nltk.FreqDist(ngram)]

    def parse_text(self, xml):
        root = ET.parse(xml)
        root = root.getroot()
        paragraphs = root.findall('PARAGRAPH')

        texts = []
        for p in paragraphs:
            ptexts = map(lambda x: nltk.word_tokenize(x.text), p.findall('STRING'))
            ptexts = reduce(lambda x,y: x+y, ptexts)
            texts.extend(ptexts)
        return texts