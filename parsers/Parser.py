__author__ = 'thiagocastroferreira'

import os

class Parser(object):
    def list_files(self, root = '../../data/xmls'):
        self.root = root
        lists = os.listdir(self.root)
        lists.remove('.DS_Store')

        dirlists = [os.path.join(self.root, l) for l in lists]
        files = [os.listdir(l) for l in dirlists]

        result = {}
        for l, f in zip(dirlists, files):
            result[l.split('/')[-1]] = []
            for fi in filter(lambda x: x != '.DS_Store', f):
                result[l.split('/')[-1]].append(os.path.join(l, fi))

        return result