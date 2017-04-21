__author__ = 'thiagocastroferreira'

import utils
import prob_counter
from collections import Counter

def original(reference):
    return utils.type2id[reference['original-type']]

def major_slot_choice(reference, references):
    pred = prob_counter.count(eqs = [{'key':'text-id', 'value': reference['text-id']}, \
                                      {'key':'reference-id', 'value': reference['reference-id']}], \
                              neqs = [{'key':'participant-id', 'value': reference['participant-id']}], \
                              dataset=references)

    pred = map(lambda x: x['type'], pred)
    pred = Counter(pred).most_common(1)[0][0]
    return utils.type2id[pred]

def major_participant_choice(reference, references):
    pred = prob_counter.count(eqs= [{'key':'participant-id', 'value': reference['participant-id']}], \
                              neqs=[], dataset=references)

    pred = map(lambda x: x['type'], pred)
    pred = Counter(pred).most_common(1)[0][0]
    return utils.type2id[pred]

def major_syntax_choice(reference, references):
    pred = prob_counter.count(eqs= [{'key':'syncat', 'value': reference['syncat']}], \
                              neqs=[], dataset=references)

    pred = map(lambda x: x['type'], pred)
    pred = Counter(pred).most_common(1)[0][0]
    return utils.type2id[pred]

def major_choice(conditions, reference, references):
    values = []
    for condition in conditions:
        values.append({'key':condition, 'value':reference[condition]})

    pred = prob_counter.count(eqs=values, neqs=[], dataset=references)
    pred = map(lambda x: x['type'], pred)
    pred = Counter(pred).most_common(1)[0][0]
    return utils.type2id[pred]