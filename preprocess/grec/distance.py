__author__ = 'thiagocastroferreira'

import os

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

'''
    Compute the paragraph position of the references and the distance from the previouse one.
    The input is the GREC texts parsed by the SentenceParser script
'''
class ScriptDistance():
    def __init__(self, root = '../../data/grec'):
        self.root_sparsed = os.path.join(root, 'sentence-parsed')
        self.root = os.path.join(root, 'oficial')
        self.files = self.list_files()

        files = self.list_files()

        for f in files:
            updated = self.update(f)

            rough_string = ET.tostring(updated, 'utf-8').replace('\n', '').replace('\t', '')
            reparsed = minidom.parseString(rough_string)
            xml = reparsed.toprettyxml(indent="\t")

            fname = os.path.join(self.root, f)
            with open(fname, 'w') as f:
                f.write(xml.encode('utf8'))

    def list_files(self):
        dirs = filter(lambda x: x != '.DS_Store', os.listdir(self.root_sparsed))
        return dirs

    def update(self, xml):
        root = ET.parse(os.path.join(self.root_sparsed, xml))
        root = root.getroot()

        context_id = root.attrib['ID']
        paragraphs = root.findall('PARAGRAPH')

        positions = []

        for paragraph in paragraphs:
            pos = 0
            paragraph_id = paragraph.attrib['ID']
            for sentence in paragraph:
                spos = 0
                sentence_id = sentence.attrib['ID']
                for child in sentence:
                    if child.tag == 'REFERENCE':
                        pos = pos + 1
                        spos = spos + 1
                        position = {'context_id':context_id, 'paragraph_id':paragraph_id,
                                    'sentence_id':sentence_id, \
                                    'slot_id': child.attrib['ID'], 'position': pos, \
                                    'sentence_position':spos}
                        paragraph_recency, sentence_recency = self.get_reference_distances(position, positions)
                        positions.append(position)

                        child.attrib['PARAGRAPH-POSITION'] = str(pos)
                        child.attrib['SENTENCE-POSITION'] = str(spos)
                        child.attrib['PARAGRAPH-RECENCY'] = str(paragraph_recency)
                        child.attrib['SENTENCE-RECENCY'] = str(sentence_recency)
                    elif child.tag == 'STRING':
                        pos = pos + len(self.clean(child.text).split())
                        spos = spos + len(self.clean(child.text).split())
        return root

    def get_reference_distances(self, position, positions):
        previous_ref = filter(lambda x: int(x['slot_id']) == int(position['slot_id'])-1 and \
                                        x['paragraph_id'] == position['paragraph_id'] and \
                                        x['context_id'] == position['context_id'], positions)
        if len(previous_ref) == 0:
            paragraph_recency = 0
        else:
            paragraph_recency = int(position['position']) - int(previous_ref[0]['position'])

        previous_ref = filter(lambda x: int(x['slot_id']) == int(position['slot_id'])-1 and \
                                        x['sentence_id'] == position['sentence_id'] and \
                                        x['context_id'] == position['context_id'], positions)
        if len(previous_ref) == 0:
            sentence_recency = 0
        else:
            sentence_recency = int(position['sentence_position']) - int(previous_ref[0]['sentence_position'])
        return paragraph_recency, sentence_recency

    def clean(self, text):
        text = text.strip()
        text = text.replace('.', '').replace('!', '').replace('?', '').replace(',', '')
        text = text.strip().replace(',', '').replace(';', '').replace(':', '')
        return text



ScriptDistance()