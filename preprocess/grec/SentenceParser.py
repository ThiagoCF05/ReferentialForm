__author__ = 'thiagocastroferreira'

import os

import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize
import xml.dom.minidom as minidom

'''
Parse the GREC texts into the format that consider sentences.

The input of this script is the texts already parsed by the Parser script
'''
class SentenceParser():
    def __init__(self, root = '../../data/grec'):
        self.root = root
        self.root_parsed = os.path.join(self.root, 'parsed')
        self.root_sparsed = os.path.join(self.root, 'sentence-parsed')
        self.files = self.list_files()

        for file in self.files:
            print file
            root = self.parse(file)

            rough_string = ET.tostring(root, 'utf-8').replace('\n', '').replace('\t', '')
            reparsed = minidom.parseString(rough_string)
            xml = reparsed.toprettyxml(indent="\t")

            fname = os.path.join(self.root_sparsed, file)
            with open(fname, 'w') as f:
                f.write(xml.encode('utf8'))

    def list_files(self):
        dirs = filter(lambda x: x != '.DS_Store', os.listdir(self.root_parsed))
        return dirs

    def parse(self, file):
        tree = ET.parse(os.path.join(self.root_parsed, file))
        root = tree.getroot()
        root.text = ''
        root.find('TITLE').tail = ''

        paragraphs = root.findall("PARAGRAPH")
        id = 1
        for paragraph in paragraphs:
            content = self.parse_paragraph(paragraph)
            paragraph.clear()
            paragraph.attrib['ID'] = str(id)
            paragraph.extend(content)
            id = id + 1
        return root

    def parse_paragraph(self, paragraph):
        text = ''
        references = []
        content = []

        for child in paragraph:
            if child.tag == 'REFERENCE':
                refname = 'REFERENCE' + str(child.attrib['ID'])
                references.append((refname, child))
                text = text + refname
            else:
                text = text + child.text.replace('\n', '') + ' '

        sentences = sent_tokenize(text)
        for s_i, s in enumerate(sentences):
            sent = ET.Element('SENTENCE')
            sent.attrib['ID'] = str(paragraph.attrib['ID']) + '.' + str(s_i+1)

            used = []
            for reference in references:
                if reference[0] in s:
                    used.append(reference[0])
                    aux = s.split(reference[0])
                    if aux[0].strip() != '':
                        string = ET.Element('STRING')
                        string.text = aux[0]
                        sent.append(string)
                    sent.append(reference[1])
                    s = aux[1]

            string = ET.Element('STRING')
            string.text = s
            sent.append(string)

            references = filter(lambda x: x[0] not in used, references)
            content.append(sent)
        return content

SentenceParser()