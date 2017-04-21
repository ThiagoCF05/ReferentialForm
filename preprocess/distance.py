__author__ = 'thiagocastroferreira'

from parsers.Parser import Parser

import xml.etree.ElementTree as ET

# Compute the paragraph position of the references and the distance from the previouse one
class ScriptDistance(Parser):
    def __init__(self):
        super(Parser)

        lists = self.list_files()
        positions = []

        keys = lists.keys()
        keys.sort()
        for l in keys:
            for f in lists[l]:
                positions.extend(self.get_reference_distances(self.get_reference_positions(f)))

        print self.parse_to_sql(positions)

    def get_reference_positions(self, xml):
        root = ET.parse(xml)
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
                        positions.append({'context_id':context_id, 'paragraph_id':paragraph_id,
                                          'sentence_id':sentence_id,\
                                          'slot_id': child.attrib['ID'], 'position': pos, \
                                          'sentence_position':spos})
                    elif child.tag == 'STRING':
                        pos = pos + len(self.clean(child.text).split())
                        spos = spos + len(self.clean(child.text).split())
        return positions

    def get_reference_distances(self, positions):
        for position in positions:
            previous_ref = filter(lambda x: int(x['slot_id']) == int(position['slot_id'])-1 and \
                             x['paragraph_id'] == position['paragraph_id'] and \
                             x['context_id'] == position['context_id'], positions)
            if len(previous_ref) == 0:
                position['paragraph_recency'] = 0
            else:
                position['paragraph_recency'] = int(position['position']) - int(previous_ref[0]['position'])

            previous_ref = filter(lambda x: int(x['slot_id']) == int(position['slot_id'])-1 and \
                                            x['sentence_id'] == position['sentence_id'] and \
                                            x['context_id'] == position['context_id'], positions)
            if len(previous_ref) == 0:
                position['sentence_recency'] = 0
            else:
                position['sentence_recency'] = int(position['sentence_position']) - int(previous_ref[0]['sentence_position'])
        return positions

    def clean(self, text):
        text = text.strip()
        text = text.replace('.', '').replace('!', '').replace('?', '').replace(',', '')
        text = text.strip().replace(',', '').replace(';', '').replace(':', '')
        return text

    def parse_to_sql(self, positions):
        sql = ''
        for position in positions:
            sql = sql + 'UPDATE `results` SET `position` = ' + str(position['position'])
            sql = sql + ', `dist_prev_reference` = ' + str(position['paragraph_recency'])
            sql = sql + ', `sentence_position` = ' + str(position['sentence_position'])
            sql = sql + ', `sentence_recency` = ' + str(position['sentence_recency'])
            sql = sql + ' WHERE `context_id` = ' + str(position['context_id'])
            sql = sql + ' AND `slot_id` = ' + str(position['slot_id']) + ';\n'
        return sql

ScriptDistance()