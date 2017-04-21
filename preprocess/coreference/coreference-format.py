# Format the xmls presented in coreferences-merged and produce a json to compute the results
__author__ = 'thiagocastroferreira'

import os
import json

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

def parse_xml(root, write_file):
    rough_string = ET.tostring(root, 'utf-8').replace('\n', '').replace('\t', '')
    reparsed = minidom.parseString(rough_string)
    xml = reparsed.toprettyxml(indent="\t")

    with open(write_file, 'w') as f:
        f.write(xml.encode('utf8'))

def parse_json(root, name):
    def parse_references(elem, paragraph_id, sentence_id):
        refs = elem.findall('REFERENCE')
        references = filter(lambda x: 'TYPE' in x.attrib.keys(), \
                            map(lambda x: x, refs))
        treferences = map(lambda x: x, elem.findall('TARGET-REFERENCE'))

        for reference in references:
            reference.attrib['PARAGRAPH-ID'] = paragraph_id
            reference.attrib['SENTENCE-ID'] = sentence_id
            reference.attrib['TEXT'] = name

        for reference in treferences:
            reference.attrib['PARAGRAPH-ID'] = paragraph_id
            reference.attrib['SENTENCE-ID'] = sentence_id
            reference.attrib['TEXT'] = name

            refexes = reference.find('PARTICIPANTS-REFEX').findall('REFEX')
            refexes = dict(map(lambda x: (x.attrib['PARTICIPANT-ID'], x.attrib['TYPE']), refexes))
            reference.attrib['PARTICIPANTS-REFEX'] = refexes

        target_ref.extend(map(lambda x: x.attrib, treferences))
        distractors_ref.extend(map(lambda x: x.attrib, references))

        for ref in refs:
            parse_references(ref, paragraph_id, sentence_id)

    target_ref = []
    distractors_ref = []
    paragraphs = root.findall('PARAGRAPH')
    for p in paragraphs:
        sentences = p.findall('SENTENCE')
        for s in sentences:
            parse_references(s, p.attrib['ID'], s.attrib['ID'])
    return target_ref, distractors_ref


def set_attributes(reference, mention_id):
    reference.attrib['ANIMACY'] = reference.attrib['animacy'].lower()
    del reference.attrib['animacy']
    reference.attrib['ENTITY-ID'] = reference.attrib['entity'].lower()
    del reference.attrib['entity']
    reference.attrib['GENDER'] = reference.attrib['gender'].lower()
    del reference.attrib['gender']
    reference.attrib['NUMBER'] = reference.attrib['number'].lower()
    del reference.attrib['number']
    reference.attrib['MENTION-ID'] = str(mention_id)
    del reference.attrib['mentionid']
    if 'mentiontype' in reference.attrib.keys():
        if reference.attrib['mentiontype'] == 'NOMINAL':
            if 'this' in reference.text.encode('utf-8').lower() \
                    or 'that' in reference.text.encode('utf-8').lower() \
                    or 'these' in reference.text.encode('utf-8').lower() \
                    or 'those' in reference.text.encode('utf-8').lower():
                reference.attrib['TYPE'] = 'demonstrative'
            else:
                reference.attrib['TYPE'] = 'description'
        elif reference.attrib['mentiontype'] == 'PROPER':
            reference.attrib['TYPE'] = 'name'
        elif reference.attrib['mentiontype'] == 'PRONOMINAL':
            reference.attrib['TYPE'] = 'pronoun'
        del reference.attrib['mentiontype']
    return reference

def parse(reference, mention_id):
    reference = set_attributes(reference, mention_id)
    mention_id = mention_id + 1
    new_tags = []
    for child in reference:
        if child.tag == 'REFERENCE':
            r, mention_id = parse(child, mention_id)
            new_tags.append(r)
        elif child.tag == 'STRING':
            new_tags.append(child)
        elif child.tag == 'TARGET-REFERENCE':
            new_tags.append(set_attributes(child, mention_id))
            mention_id = mention_id + 1

    reference._children = []
    reference.extend(new_tags)
    return reference, mention_id

def parse_text(read_file):
    tree = ET.parse(read_file)
    root = tree.getroot()

    paragraphs = root.findall('PARAGRAPH')
    mention_id = 1
    for p_i, paragraph in enumerate(paragraphs):
        sentences = paragraph.findall('SENTENCE')
        for s_i, s in enumerate(sentences):
            s.attrib['ID'] = str(p_i+1) + '.' + str(s_i+1)

            new_tags = []
            for child in s:
                if child.tag == 'REFERENCE':
                    r, mention_id = parse(child, mention_id)
                    new_tags.append(r)
                elif child.tag == 'STRING':
                    new_tags.append(child)
                elif child.tag == 'TARGET-REFERENCE':
                    new_tags.append(set_attributes(child, mention_id))
                    mention_id = mention_id + 1

            s._children = []
            s.extend(new_tags)

    return root

read = '../data/coreferences-merged'
write = '../data/coreferences-oficial'
dirs = os.listdir(read)
dirs.remove('.DS_Store')

target_references, distractor_references = [], []
for dir in dirs:
    read_path = os.path.join(read, dir)
    write_path = os.path.join(write, dir)
    files = os.listdir(read_path)
    files.remove('.DS_Store')
    for file in files:
        print file
        root = parse_text(os.path.join(read_path, file))
        parse_xml(root, os.path.join(write_path, file))
        t,d = parse_json(root, dir + '/' +file)
        target_references.extend(t)
        distractor_references.extend(d)

        results = {'target':target_references, 'distractor': distractor_references}
        json.dump(results, open('analysis.json', 'w'))
