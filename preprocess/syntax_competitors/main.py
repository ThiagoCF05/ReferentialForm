__author__ = 'thiagocastroferreira'

from parsers.TextParser import TextParser
from stanford_corenlp_pywrapper import CoreNLP
from syntax import Syntax

# text = '(ROOT (S (NP (NNP Philip) (NNP Frederick) (NNP Anschutz)) (VP (VBD announced) (NP-TMP (NNP Tuesday)) (SBAR (IN that) (S (NP (PRP he)) (VP (VBD was) (VP (VBG retiring) (PP (IN from) (NP (NP (DT the) (NNS boards)) (PP (IN of) (NP (NP (CD three) (NNS companies)) (SBAR (WHPP (IN in) (WHNP (WDT which))) (S (NP (PRP he)) (VP (VBZ is) (NP (NP (DT the) (JJS largest) (NN shareholder)) (: :) (NP (NP (NNP Union) (NNP Pacific) (NNP Corp.)) (, ,) (NP (NNP Regal) (NNP Entertainment) (NNP Group)) (CC and) (NP (NNP Qwest) (NNPS Communications) (NNP International) (NNP Inc.)))))))))))))))) (. .)))'

def parse(text):
    text = text[9:-2]

    node = {'type':'syntax', 'tag':'ROOT', 'token':'', 'phrase':'', 'reference':0, 'next':[]}

    tree = {'root': node}
    stack = ['root']
    _tag = ''
    _type = 'syntax'
    for i in range(len(text)):
        if text[i] == '(':
            _tag = ''
        elif text[i] == ' ':
            if text[i+1] == '(' and text[i-1] != ')':
                _type = 'syntax'
                _id = len(tree.keys())+1
                tree[_id] = {'type':_type, 'tag':_tag, 'token':'', 'phrase':'', 'reference':0, 'next':[]}
                tree[stack[-1]]['next'].append(_id)
                stack.append(_id)
            elif text[i+1] != '(' and text[i-1] != ')':
                _type = 'pos_tag'
                stack.append(_type)
                _tag = _tag + ' '
        elif text[i] == ')':
            if stack[-1] == 'pos_tag':
                stack.pop()
                _id = len(tree.keys())+1
                tag, token = _tag.split()
                tree[_id] = {'type':_type, 'tag':tag, 'token':token, 'reference':0, 'next':[]}
                tree[stack[-1]]['next'].append(_id)

                for node in stack:
                    if tree[node]['tag'] == 'NP':
                        tree[node]['phrase'] = tree[node]['phrase'] + ' ' + token
            else:
                stack.pop()
        else:
            _tag = _tag + text[i]
    return tree

if __name__ == '__main__':
    proc = CoreNLP("coref")

    trees = {}
    for e in [TextParser().run()[11]]:
        list_id, id, text, originals = e
        trees[id] = {}
        for paragraph in text:
            r = proc.parse_doc(text[paragraph])
            trees[id][paragraph] = []
            for s_i, sentence in enumerate(r['sentences']):
                tree = parse(sentence['parse'])
                trees[id][paragraph].append(tree)
                # for node in tree:
                #     print node, tree[node]
                # print '\n'

                references = filter(lambda x: x['paragraph_id'] == paragraph and x['sentence_id'] == s_i+1, originals)
                s = Syntax(tree, references)
                print sentence['parse']
                print references
                print s.types
                print '\n'

