__author__ = 'thiagocastroferreira'

# Compute the intra-variation of each referential form in each slot
# Distinguish the classes by the levenshtein distance measure
# If the distance among 2 references is greater than 2, they are classified in different classes

from parsers.ReferenceParser import ReferenceParser
import utils
import copy
import numpy as np

from nltk.metrics.distance import edit_distance

def normalized_entropy(events):
    entropy = 0.0
    for event in events:
        if events[event] != 0.0:
            if len(events.keys()) == 1:
                entropy = 0.0
            else:
                entropy = entropy + ((events[event] * np.log(events[event])) / np.log(len(events.keys())))
    return -1 * entropy

# Calculate probability distribution among the references of the same type in a slot
def get_distribution(events, references):
    distribution = dict(map(lambda x: (tuple(x), 0.0), events))
    references_by_slot_type = map(lambda x: x['refex'].split('\'')[0].lower().strip(), filter(lambda x: x['text-id'] == str(text) \
                                                                                                        and x['reference-id'] == str(slot) \
                                                                                                        and x['type'] == str(type), references))

    for reference in references_by_slot_type:
        t = filter(lambda x: reference in x, distribution.keys())[0]
        distribution[t] = distribution[t] + 1

    denominator = sum(distribution.values())
    for k,v in distribution.iteritems():
        distribution[k] = v / denominator
    return distribution

# Group the references by text, slot and type to facilitate the intra-variation calculus
def group_by_text(references):
    types = utils.type2id
    del types['other']
    del types['empty']
    types = types.keys()

    texts = list(set(map(lambda x: int(x['text-id']), references)))
    texts.sort()

    groups = {}

    for text in texts:
        groups[text] = {}
        filtered = filter(lambda x: x['text-id'] == str(text), references)
        slots = list(set(map(lambda y: int(y['reference-id']), filtered)))
        slots.sort()
        for slot in slots:
            groups[text][slot] = {}
            for type in types:
                groups[text][slot][type] = list(set(map(lambda x: x['refex'].split('\'')[0].lower().strip(), filter(lambda x: x['type'] == type, filtered))))
    return groups

if __name__ == '__main__':
    verbose = True

    parser = ReferenceParser()
    references = parser.run()
    del parser

    groups = group_by_text(references)

    for text in groups:
        for slot in groups[text]:
            for type in groups[text][slot]:
                clusters = []
                while len(groups[text][slot][type]) != 0:
                    s1 = groups[text][slot][type][0]
                    del groups[text][slot][type][0]
                    cluster = [s1]

                    aux = copy.copy(groups[text][slot][type])
                    for i in range(len(groups[text][slot][type])):
                        s2 = groups[text][slot][type][i]
                        dist = edit_distance(s1.strip(), s2.strip())
                        if dist <= 2:
                            cluster.append(s2)
                            del aux[aux.index(s2)]
                    clusters.append(cluster)
                    groups[text][slot][type] = aux

                # distribution = get_distribution(clusters, references)

                if verbose:
                    print 'Text:', text, 'Slot:', slot, 'Type:', type
                    print 'Type: ', type
                    print 'References: '
                    for cluster in clusters:
                        print '\t', cluster
                    # print 'Distribution: '
                    # for k, v in distribution.iteritems():
                    #     print '\t', list(k), round(v, 4)
                    # print 'Entropy: ', round(normalized_entropy(distribution), 4)
                    print 50 * '-'
                    # pass
                else:
                    if type == 'name':
                        t = 'n'
                    elif type == 'pronoun':
                        t = 'pro'
                    elif type == 'description':
                        t = 'des'
                    else:
                        t = 'dem'

                    sql = 'INSERT INTO intra_results (entropy, type, context_id, slot_id) VALUES ('
                    sql = sql + str(round(normalized_entropy(distribution), 6)) + ', '
                    sql = sql + '\'' + t + '\', '
                    sql = sql + str(text) + ', '
                    sql = sql + str(slot) + ');'
                    print sql
