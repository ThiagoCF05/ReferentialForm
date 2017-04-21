# Merge the json coreference representation with the collected references
__author__ = 'thiagocastroferreira'

import json
import os

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

class Coreference(object):
    def __init__(self):
        dir = '../data/coreferences-json'
        files = os.listdir(dir)
        files.remove('.DS_Store')
        for file in files:
            print file
            self.parse_text(os.path.join(dir, file), file)


    def parse_text(self, file, fname):
        self.root = ET.fromstring("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<TEXT></TEXT>")

        with open(file) as f:
            doc = f.read()
        self.text = json.loads(doc)

        mentions = []
        for entity in self.text['entities']:
            for mention in entity['mentions']:
                mention['entity'] = entity['entityid']
                mention['size'] = mention['tokspan_in_sentence'][1] - mention['tokspan_in_sentence'][0]
                mentions.append(mention)

        for s_i, s in enumerate(self.text['sentences']):
            aux = self.mark_references(s, filter(lambda x: x['sentence'] == s_i, mentions), 0, len(s['tokens']))
            sentence = ET.SubElement(self.root, 'SENTENCE')
            sentence.attrib['ID'] = str(s_i+1)
            sentence.extend(aux)

        rough_string = ET.tostring(self.root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        xml = reparsed.toprettyxml(indent="\t")

        fname = fname.split('.')[0] + '.xml'
        fname = os.path.join('../data/coreferences', fname)
        with open(fname, 'w') as f:
            f.write(xml.encode('utf8'))


    def mark_references(self, sentence, mentions, begin, end):
        mentions.sort(key=lambda x: (x['size'], -x['tokspan_in_sentence'][0]))

        xmls = {}

        while len(mentions) != 0:
            mention = mentions.pop()

            elem = ET.Element('REFERENCE')
            elem.attrib['animacy'] = mention['animacy']
            elem.attrib['gender'] = mention['gender']
            elem.attrib['mentiontype'] = mention['mentiontype']
            elem.attrib['number'] = mention['number']
            elem.attrib['entity'] = str(mention['entity'])
            elem.attrib['mentionid'] = str(mention['mentionid'])

            filtered = filter(lambda x: x['tokspan_in_sentence'][0] >= mention['tokspan_in_sentence'][0] \
                                        and x['tokspan_in_sentence'][1] <= mention['tokspan_in_sentence'][1], mentions)
            if len(filtered) > 0:
                elem.extend(self.mark_references(sentence, filtered, mention['tokspan_in_sentence'][0], mention['tokspan_in_sentence'][1]))
            else:
                elem.text = self.merge_sentence(sentence['tokens'][mention['tokspan_in_sentence'][0]:mention['tokspan_in_sentence'][1]], \
                                                sentence['pos'][mention['tokspan_in_sentence'][0]:mention['tokspan_in_sentence'][1]])
            xmls[(mention['tokspan_in_sentence'][0], mention['tokspan_in_sentence'][1])] = elem
            mentions = filter(lambda x: x['tokspan_in_sentence'][1] < mention['tokspan_in_sentence'][0] \
                                        or x['tokspan_in_sentence'][0] > mention['tokspan_in_sentence'][1], mentions)
        keys = xmls.keys()
        keys.sort(key=lambda x: x[0])
        results = []

        for i in range(len(keys)):
            if i == 0:
                if keys[0][0] > begin:
                    elem = ET.Element('STRING')
                    elem.text = self.merge_sentence(sentence['tokens'][begin:keys[0][0]], \
                                                    sentence['pos'][begin:keys[0][0]])
                    results.append(elem)
            else:
                elem = ET.Element('STRING')
                elem.text = self.merge_sentence(sentence['tokens'][keys[i-1][1]:keys[i][0]], \
                                                sentence['pos'][keys[i-1][1]:keys[i][0]])
                results.append(elem)
            results.append(xmls[keys[i]])
        if len(keys) > 0:
            if keys[-1][1] < end:
                elem = ET.Element('STRING')
                elem.text = self.merge_sentence(sentence['tokens'][keys[-1][1]:end], \
                                                sentence['pos'][keys[-1][1]:end])
                results.append(elem)
        return results

    def merge_sentence(self, sentence, pos):
        s = ' '
        for i, tag in enumerate(pos):
            if tag in ['.', ',', ':', ';', '!', '?', 'POS', '\'', '\'\'', '``']:
                s = s.rstrip() + sentence[i] + ' '
            elif tag in ['``']:
                s = s + sentence[i]
            else:
                s = s + sentence[i] + ' '
        return s

coref = Coreference()