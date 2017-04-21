__author__ = 'thiagocastroferreira'

import os

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

'''
Parse the original GREC texts to the format used in the VarREG corpus
'''
class Parser():
    def __init__(self, root = '../../data/grec'):
        self.root = root
        self.root_raw = os.path.join(self.root, 'originals')
        self.root_parsed = os.path.join(self.root, 'parsed')
        self.files = self.list_files()

        for file in self.files:
            print file
            root = self.parse(file)

            rough_string = ET.tostring(root, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            xml = reparsed.toprettyxml(indent="\t")

            fname = os.path.join(self.root_parsed, file)
            with open(fname, 'w') as f:
                f.write(xml.encode('utf8'))

    def list_files(self):
        dirs = filter(lambda x: x != '.DS_Store', os.listdir(self.root_raw))
        return dirs

    def parse(self, file):
        tree = ET.parse(os.path.join(self.root_raw, file))
        root = tree.getroot()
        root.text = ''
        root.find('TITLE').tail = ''

        paragraphs = root.findall("PARAGRAPH")
        id = 1
        reference_id = 1
        for paragraph in paragraphs:
            reference_id, subelements = self.parse_paragraph(paragraph, reference_id)
            paragraph.clear()
            paragraph.attrib['ID'] = str(id)
            paragraph.extend(subelements)
            id = id + 1
        return root

    def parse_paragraph(self, paragraph, reference_id):
        text = list(paragraph.itertext())
        text = filter(lambda x: x != "\n", text)
        text = map(lambda x: x.replace("\n", ""), text)

        references = paragraph.findall("REF")
        parsed_references = []
        for reference in references:
            parsed_references.append(self.parse_reference_to_xml(reference, reference_id))
            reference_id = reference_id + 1

        parsed_references = iter(parsed_references)
        references = iter(map(lambda reference: self.parse_reference(reference), references))

        content = []
        pos = 0
        try:
            reference = references.next()
            parsed_reference = parsed_references.next()
        except Exception, e:
            reference = { "REFEX": {"TEXT": ""} }
        while pos < len(text):
            if reference["REFEX"]["TEXT"].lower() == text[pos].lower():
                content.append(parsed_reference)
                pos = pos + len(reference["ALT-REFEX"]) + 1
                try:
                    reference = references.next()
                    parsed_reference = parsed_references.next()
                except Exception, e:
                    reference = { "REFEX": {"TEXT": ""} }
            else:
                string = ET.Element('STRING')
                string.text = text[pos]
                content.append(string)
                pos = pos + 1
        return reference_id, content

    def parse_reference(self, reference):
        def parse_alt_refex(refex):
            expression = refex.attrib
            expression["TEXT"] = refex.text
            return expression

        ref = reference.attrib

        refex = reference.find("REFEX")
        ref["REFEX"] = refex.attrib
        ref["REFEX"]["TEXT"] = refex.text

        alt_refex = reference.find("ALT-REFEX").findall("REFEX")

        ref["ALT-REFEX"] = map(lambda refex: parse_alt_refex(refex), alt_refex)
        return ref

    def parse_reference_to_xml(self, reference, id):
        parsed = ET.Element('REFERENCE')
        parsed.attrib['ID'] = str(id)
        parsed.attrib['SYNCAT'] = reference.attrib['SYNCAT']
        parsed.attrib['SEMCAT'] = reference.attrib['SEMCAT']


        aux = reference.find("REFEX")
        original = ET.SubElement(parsed, 'ORIGINAL-REFEX')
        refex = ET.SubElement(original, 'REFEX')
        refex.text = aux.text

        try:
            refex.attrib['EMPHATIC'] = aux.attrib['EMPHATIC']
            refex.attrib['HEAD'] = aux.attrib['HEAD']
            refex.attrib['CASE'] = aux.attrib['CASE']
        except:
            refex.attrib['EMPHATIC'] = ''
            refex.attrib['HEAD'] = ''
            refex.attrib['CASE'] = ''

        if aux.attrib['REG08-TYPE'] == 'empty':
            refex.attrib['TYPE'] = 'empty'
        else:
            if aux.attrib['EMPHATIC'] == 'yes':
                refex.attrib['TYPE'] = 'pronoun'
            elif aux.attrib['REG08-TYPE'] == 'common':
                determiner = aux.text.split()[0]
                if "this" in determiner or "that" in determiner or \
                                "these" in determiner or "those" in determiner:
                    refex.attrib['TYPE'] = 'demonstrative'
                else:
                    refex.attrib['TYPE'] = 'description'
            else:
                refex.attrib['TYPE'] = aux.attrib['REG08-TYPE']
        return parsed

Parser()