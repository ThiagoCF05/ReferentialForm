__author__ = 'thiagocastroferreira'

class Realizer(object):
    def __init__(self, references, forms, expressions):
        self.references = references
        self.forms = forms
        self.expressions = expressions

    def run(self):
        results = []
        for reference in self.references:
            form = filter(lambda x: x[0] == reference['reference-id'], self.forms)[0]

            if form[1] in ['description', 'demonstrative']:
                type = 'common'
            else:
                type = form[1]

            if type == 'common':
                results.append(self.realize_common(reference, form[1]))
            elif type == 'empty':
                results.append((reference['reference-id'], '_'))
            elif type == 'pronoun':
                results.append(self.realize_pronoun(reference))
            elif type == 'name':
                results.append(self.realize_name(reference))
        return results

    def realize_pronoun(self, reference):
        if reference['head'] == 'rel-pron':
            if reference['syncat'] == 'subj-det':
                f = filter(lambda x: x['head'] == 'rel-pron' \
                                     and x['case'] == 'genitive' \
                                     and x['emphatic'] == reference['emphatic'] \
                                     and x['type'] == 'pronoun', self.expressions)
            else:
                f = filter(lambda x: x['head'] == 'rel-pron' \
                                     and x['case'] == 'nominative' \
                                     and x['emphatic'] == reference['emphatic'] \
                                     and x['type'] == 'pronoun', self.expressions)
        else:
            if reference['syncat'] == 'subj-det':
                f = filter(lambda x: x['head'] == 'pronoun' \
                                     and x['case'] == 'genitive' \
                                     and x['emphatic'] == reference['emphatic'] \
                                     and x['type'] == 'pronoun', self.expressions)
            elif reference['syncat'] == 'np-subj':
                f = filter(lambda x: x['head'] == 'pronoun' \
                                     and x['case'] == 'nominative' \
                                     and x['emphatic'] == reference['emphatic'] \
                                     and x['type'] == 'pronoun', self.expressions)
            else:
                f = filter(lambda x: x['head'] == 'pronoun' \
                                                and x['case'] == 'accusative' \
                                                and x['emphatic'] == reference['emphatic'] \
                                                and x['type'] == 'pronoun', self.expressions)
        return reference['reference-id'], f[0]['text']

    def realize_common(self, reference, form):
        if reference['syncat'] == 'subj-det':
            f = filter(lambda x: x['emphatic'] == reference['emphatic'] \
                                 and x['head'] == 'nominal' \
                                 and x['case'] == 'genitive' \
                                 and x['type'] == 'common', self.expressions)
        else:
            f = filter(lambda x: x['emphatic'] == reference['emphatic'] \
                                 and x['head'] == 'nominal' \
                                 and x['case'] == 'plain' \
                                 and x['type'] == 'common', self.expressions)
        if form == 'description':
            return reference['reference-id'], f[0]['text']
        else:
            expression = "this" + f[0]['text'][3:]
            return reference['reference-id'], expression

    def realize_name(self, reference):
        if reference['syncat'] == 'subj-det':
            f = filter(lambda x: x['case'] == 'genitive' \
                                 and x['head'] == 'nominal' \
                                 and x['emphatic'] == reference['emphatic'] \
                                 and x['type'] == 'name', self.expressions)
        else:
            f = filter(lambda x: x['case'] == 'plain' \
                                 and x['head'] == 'nominal' \
                                 and x['emphatic'] == reference['emphatic'] \
                                 and x['type'] == 'name', self.expressions)
        sizes = map(lambda x: len(x['text']), f)
        if reference['givenness'] == 'new':
            return reference['reference-id'], filter(lambda x: len(x['text']) == max(sizes), f)[0]['text']
        else:
            return reference['reference-id'], filter(lambda x: len(x['text']) == min(sizes), f)[0]['text']