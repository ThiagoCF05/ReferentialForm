from experiment_model.Realizer import Realizer

__author__ = 'thiagocastroferreira'

from models.naive_bayes import NaiveBayes
from models.random_model import RandomModel
from parsers.ReferenceParser import ReferenceParser
from parsers.grec.Parser import GRECParser
from random import shuffle

import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

# import cPickle as p
import json
import utils
import os

class Model(object):
    def __init__(self, features, classes):
        parser = ReferenceParser()
        self.trainset = parser.run()
        self.features = features
        self.classes = classes
        self.model = self.__model__()
        self.random_model = RandomModel(self.classes)

        self.descriptions = json.load(open('../data/grec/descriptions.json'))

    def __model__(self):
        model = NaiveBayes(self.trainset, self.features, self.classes)
        model.train()
        return model

    def run(self):
        root = '../data/grec/oficial'
        self.parser = GRECParser(root)

        texts = filter(lambda x: x != '.DS_Store', os.listdir(root))
        texts.reverse()
        prev = ''
        for text in texts:
            prev = self.proc_text(text, prev)

    def proc_text(self, text, prev):
        # Parse references from the text
        references = self.parser.parse_text(text)
        if len(references) == 10 and text.split()[-1] != '2.xml':
            print text
            # Get all possible referring expressions possible for the references of the text
            expressions = self.get_expressions(text)
            # Group the references according to the features that describe them and compute their referential form distributions
            groups = self.group(references)
            
            forms = reduce(lambda x,y: x+y,[self.choose_form(groups[group]['references'], groups[group]['distribution']) for group in groups])
            regenerated = Realizer(references, forms, expressions).run()
            randomized = Realizer(references, self.choose_form(references, self.random_model.classify()), expressions).run()
            originals = map(lambda x: (int(x['reference-id']), x['refex']), references)
            
            self.parse_report(text, randomized, regenerated, originals, prev, 'report')
            
            self.parse_html(text, randomized, prev, 'html/random')
            self.parse_html(text, originals, prev, 'html/original')
            return self.parse_html(text, regenerated, prev, 'html/regenerated')
        return prev

    def get_expressions(self, text):
        root = ET.parse(os.path.join('../data/grec/originals', text))
        root = root.getroot()

        expressions = []
        reference = None
        for paragraph in root.findall('PARAGRAPH'):
            references = paragraph.findall('REF')
            for reference in references:
                _expressions = reference.find('ALT-REFEX').findall('REFEX')

                for _expression in _expressions:
                    expression = {}
                    try:
                        expression['type'] = _expression.attrib['REG08-TYPE']
                        expression['emphatic'] = _expression.attrib['EMPHATIC']
                        expression['head'] = _expression.attrib['HEAD']
                        expression['case'] = _expression.attrib['CASE']
                        expression['text'] = _expression.text
                    except:
                        expression['type'] = _expression.attrib['REG08-TYPE']
                        expression['emphatic'] = ''
                        expression['head'] = ''
                        expression['case'] = ''
                        expression['text'] = _expression.text

                    if expression not in expressions:
                        expressions.append(expression)

        if reference != None and reference.attrib['SEMCAT'] == 'person':
            description = filter(lambda x: x['text'] == text, self.descriptions)[0]['description']
            expressions = self.define_description(description, expressions)

        return expressions

    def define_description(self, description, expressions):
        expression = {'type':'common', 'head':'nominal', 'case':'plain', 'emphatic':'no'}
        expression['text'] = description
        expressions.append(expression)

        nominal = filter(lambda x: x['type'] == 'name' \
                                   and x['head'] == 'nominal' \
                                   and x['case'] == 'plain' \
                                   and x['emphatic'] == 'yes', expressions)[0]
        expression = {'type':'common', 'head':'nominal', 'case':'plain', 'emphatic':'yes'}
        if 'himself' in nominal['text']:
            expression['text'] = description + ' himself'
        else:
            expression['text'] = description + ' herself'
        expressions.append(expression)

        expression = {'type':'common', 'head':'nominal', 'case':'genitive', 'emphatic':'no'}
        expression['text'] = description + '\'s'
        expressions.append(expression)

        return expressions

    def group(self, references):
        g = {}
        for reference in references:
            X = dict([(feature, reference[feature]) for feature in self.features.keys()])
            if tuple(X.values()) not in g:
                g[tuple(X.values())] = {'distribution': self.model.classify(X, 'minimun_float'), 'references':[]}

            _reference = {}
            _reference['reference-id'] = reference['reference-id']
            _reference['sentence-recency'] = reference['sentence-recency']
            _reference['syncat'] = reference['syncat']
            _reference['givenness'] = reference['givenness']
            _reference['head'] = reference['head']
            _reference['case'] = reference['case']
            _reference['emphatic'] = reference['head']
            _reference['type'] = reference['type']
            _reference['refex'] = reference['refex']

            g[tuple(X.values())]['references'].append(_reference)
        return g

    def choose_form(self, references, distribution):
        size = len(references)

        'Remove relational pronouns from the choice based on variation'
        result = map(lambda reference: (reference['reference-id'], reference['type']), filter(lambda x: x['head'] == 'rel-pron', references))
        size -= len(result)
        'Remove empty references from the choice based on variation'
        empty = map(lambda reference: (reference['reference-id'], reference['type']), filter(lambda x: x['type'] == 'empty', references))
        size -= len(empty)
        result.extend(empty)

        for form in distribution:
            distribution[form] = size * distribution[form]

        # print distribution
        _references = filter(lambda x: x['head'] != 'rel-pron' and x['type'] != 'empty', references)
        shuffle(_references)
        for reference in _references:
            form = filter(lambda x: distribution[x] == max(distribution.values()), distribution.keys())[0]
            result.append((reference['reference-id'], form))

            distribution[form] -= 1
        # print result
        # print 10 * '-'
        return result

    def parse_report(self, text, randomized, regenerated, originals, prev_text, directory):
        root = ET.parse(os.path.join('../data/grec/oficial', text))
        root = root.getroot()
        title = root.find('TITLE').text

        soup = BeautifulSoup(open('html/report.html'))

        if prev_text != '':
            link = soup.find('a', {"id":"next"})
            link['href'] = prev_text + ".html"

        soup.title.append(title)

        soup.body.header.h1.append(title)

        paragraphs = root.findall('PARAGRAPH')
        reference_id = 1

        original_div = soup.find('div', {'id':'original'})
        random_div = soup.find('div', {'id':'random'})
        regenerated_div = soup.find('div', {'id':'regenerated'})

        for p_i, p in enumerate(paragraphs):
            original_p = BeautifulSoup("<p class=\"lead\"></p>").p
            random_p = BeautifulSoup("<p class=\"lead\"></p>").p
            regenerated_p = BeautifulSoup("<p class=\"lead\"></p>").p

            sentences = p.findall('SENTENCE')
            for s_i, s in enumerate(sentences):
                new_sentence = True
                for i, child in enumerate(s):
                    if child.tag == 'REFERENCE':
                        original = filter(lambda x: int(x[0]) == reference_id, originals)[0][1]
                        reg = filter(lambda x: int(x[0]) == reference_id, regenerated)[0][1]
                        rand = filter(lambda x: int(x[0]) == reference_id, randomized)[0][1]
                        if new_sentence:
                            new_sentence = False
                            try:
                                original = original[0].upper() + original[1:]
                                reg = reg[0].upper() + reg[1:]
                                rand = rand[0].upper() + rand[1:]
                            except:
                                pass

                        tag_span = BeautifulSoup("<span style=\"background-color: #FFFF00\">"+ original.strip()+"</span>").span
                        original_p.append(tag_span)

                        tag_span = BeautifulSoup("<span style=\"background-color: #FFFF00\">"+ reg.strip()+"</span>").span
                        regenerated_p.append(tag_span)

                        tag_span = BeautifulSoup("<span style=\"background-color: #FFFF00\">"+ rand.strip()+"</span>").span
                        random_p.append(tag_span)
                        reference_id += 1
                    elif child.tag == 'STRING':
                        if new_sentence:
                            new_sentence = False

                        original_p.append(child.text)
                        regenerated_p.append(child.text)
                        random_p.append(child.text)
            original_div.append(original_p)
            regenerated_div.append(regenerated_p)
            random_div.append(random_p)

        f = open(os.path.join(directory, title + ".html"), 'w')
        f.write(soup.prettify(formatter="html").encode("utf-8"))
        f.close()

    def parse_html(self, text, expressions, prev_text='', directory='html/'):
        root = ET.parse(os.path.join('../data/grec/oficial', text))
        root = root.getroot()
        title = root.find('TITLE').text

        soup = BeautifulSoup(open('html/layout.html'))

        if prev_text != '':
            link = soup.find('a', {"id":"next"})
            link['href'] = prev_text + ".html"

        soup.title.append(title)

        soup.body.header.h1.append(title)

        paragraphs = root.findall('PARAGRAPH')

        reference_id = 1

        div = soup.body.div
        for p_i, p in enumerate(paragraphs):
            tag_p = BeautifulSoup("<p class=\"lead\"></p>").p

            sentences = p.findall('SENTENCE')
            for s_i, s in enumerate(sentences):
                new_sentence = True
                for i, child in enumerate(s):
                    if child.tag == 'REFERENCE':
                        expression = filter(lambda x: int(x[0]) == reference_id, expressions)[0][1]
                        if new_sentence:
                            new_sentence = False
                            try:
                                expression = expression[0].upper() + expression[1:]
                            except:
                                pass

                        tag_span = BeautifulSoup("<span style=\"background-color: #FFFF00\">"+ expression.strip()+"</span>").span
                        tag_p.append(tag_span)
                        reference_id += 1
                    elif child.tag == 'STRING':
                        if new_sentence:
                            new_sentence = False

                        tag_p.append(child.text)
            div.append(tag_p)

        f = open(os.path.join(directory, title + ".html"), 'w')
        f.write(soup.prettify(formatter="html").encode("utf-8"))
        f.close()

        return title

if __name__ == '__main__':
    parser = ReferenceParser()
    trainset = parser.run()
    testset = parser.run()
    del parser

    classes = ['name', 'pronoun', 'description', 'demonstrative']
    features = {'syncat': utils.syntax2id.keys(), \
                'givenness': utils.givenness2id.keys(), \
                'paragraph-givenness': utils.givenness2id.keys(), \
                'sentence-givenness': utils.givenness2id.keys()}
                # 'pos-bigram': utils.pos2id.keys()}

    model = Model(features, classes)
    model.run()

    # model = NaiveBayes(trainset, features, classes)
    # model.train()
    #
    # distributions = {}
    # combinations = []
    # for syntax in features['syncat']:
    #     for text_status in features['givenness']:
    #         for snt_status in features['sentence-givenness']:
    #             if (syntax, text_status, snt_status) not in combinations:
    #                 X = {'syncat':syntax, 'givenness':text_status, 'sentence-givenness':snt_status}
    #                 prob = model.classify(X)
    #
    #                 distributions[(syntax, text_status, snt_status)] = prob
    # p.dump(distributions, open('form_distributions.cPickle', 'w'))