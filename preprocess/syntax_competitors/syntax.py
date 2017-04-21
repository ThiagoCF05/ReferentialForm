__author__ = 'thiagocastroferreira'

class Syntax(object):
    def __init__(self, tree, references):
        self.types = []
        self.tree = tree
        self.references = references

        for reference in self.references:
            self.reference = reference
            self.type, self.aux_type = '', ''
            self.visited = []
            self.isFound = False
            self.clause_type('root')
            self.types.append(self.type)

    def clause_type(self, node):
        self.visited.append(node)
        if self.tree[node]['tag'] == 'SBAR':
            self.aux_type = 'subordinate'
        elif self.tree[node]['tag'] == 'S' and self.aux_type != 'subordinate' and len(self.visited) > 2:
            self.aux_type = 'coordinate'

        if self.tree[node]['tag'] == 'NP' and self.reference['reference'].strip() == self.tree[node]['phrase'].strip():
            self.isFound = True
            self.tree[node]['phrase'] = ' '
            self.type = self.aux_type

        for child in self.tree[node]['next']:
            if self.isFound:
                break
            elif child not in self.visited:
                self.clause_type(child)

        if self.tree[node]['tag'] in ['SBAR', 'S']:
            self.aux_type = ''