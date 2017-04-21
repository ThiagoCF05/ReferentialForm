__author__ = 'thiagocastroferreira'

type2id = {'name':0, 'pronoun':1, 'description':2, 'demonstrative':3, 'empty':4, 'other':5}

id2type = {0:'name', 1:'pronoun', 2:'description', 3:'demonstrative', 4:'empty', 5:'other'}

typeid2binary = {0:1, 1:0, 2:1, 3:1, 4:0}
type2binary = {'name':'long', 'pronoun':'short', 'description':'long', 'demonstrative':'long', 'empty':'short'}

pos2id = {'VBG':0, 'VBD':1, 'VBN':2, "''":3, 'VBP':4, 'WDT':5, 'JJ':6, 'VBZ':7, 'DT':8, 'NN':9, '*':10, ',':11,\
          '.':12, 'TO':13, 'RB':14, ':':15, 'NNS':16, 'NNP':17, 'VB':18, 'WRB':19, 'CC':20, 'IN':21, 'JJR':22, '-NONE-':23, 'RP':24, 'CD':25}

syntax2id = {'np-subj':0, 'np-obj':1, 'subj-det':2}
id2syntax = {0:'np-subj', 1:'np-obj', 2:'subj-det'}

clause2id = {'regular':0, 'coordinate':1, 'subordinate':2}

references = {1:'Philip', 2:'Randy Cunningham', 3:'Google', 5:'Ticktock', \
              6:'the movie', 7:'the system', 8:'Aconcagua', 9:'The Amazon River', \
              10:'Brazil', 11:'Clooney', 12:'Lawrence', 13:'Vladimir', 14:'the book', \
              15:'the movie', 16:'the phone', 17:'Cho Oyu', 18:'The Yellow River', \
              19:'Ethiopia', 21:'Dunst', 22:'Jeanine', 23:'Suncom', 24:'Airframe', \
              25:'Gothika', 26:'the phone', 27:'Kangchenjunga', 28:'Francisco River', \
              29:'Indonesia', 30:'Samuel', 31:'Huffman', 32:'Hamas', 33:'the book', \
              34:'the movie', 35:'the phone', 36:'Makalu', 37:'The Nile', 38:'India'}

topic2id = {'true': 1, 'false': 0}
competitor2id = {'1': 1, '0': 0}
distractor2id = {'true': 1, 'false': 0}
animacy2id = {'animate': 1, 'inanimate': 0}

genres2id = {'news':0, 'review':1, 'wiki':2}
givenness2id = {'new':0, 'given':1}

recency2id = {'d<=10':0, '10<d<=20':1, '20<d<=30':2, '30<d<=40':3, '40<d<=50':4, 'd>50':5}
def recency(position):
    if position <= 10:
        return 'd<=10'
    elif 10 < position <= 20:
        return '10<d<=20'
    elif 20 < position <= 30:
        return '20<d<=30'
    elif 30 < position <= 40:
        return '30<d<=40'
    elif 40 < position <= 50:
        return '40<d<=50'
    else:
        return 'd>50'

tagset = ['PRP$', 'VBG', 'VBD', '``', 'VBN', ',', "''", 'VBP', 'WDT', 'JJ', 'WP', 'VBZ', 'DT', 'RP', '$', 'NN', ')',\
          '(', 'FW', 'POS', '.', 'TO', 'LS', 'RB', ':', 'NNS', 'NNP', 'VB', 'WRB', 'CC', 'PDT', 'RBS', 'RBR', 'CD',\
          'PRP', 'EX', 'IN', 'WP$', 'MD', 'NNPS', '--', 'JJS', 'JJR', 'SYM', 'UH', '-NONE-', '#']

vareg_gain = {'pos-bigram': 0.1232565154779908, \
              'paragraph-givenness': 0.10002320150407906, \
              'sentence-givenness': 0.091307977238965055, \
              'pos-trigram': 0.069787093786886009, \
              '+pos-bigram': 0.052320854872558054, \
              '+pos-trigram': 0.049432347556003134, \
              'givenness': 0.046485351405974873, \
              'syncat': 0.046123149778272143, \
              'animacy': 0.026886793218769633, \
              'categorical-recency': 0.018420654577016714}

grec_gain = {'pos-bigram': 0.34077512590557579, \
             'pos-trigram': 0.17737026335818148, \
             'paragraph-givenness': 0.17679524075013997, \
             'sentence-givenness': 0.166219846604933, \
             'givenness': 0.14964114628019165, \
             'syncat': 0.13081054254671876, \
             '+pos-bigram': 0.10598385202442864, \
             '+pos-trigram': 0.077905400669210215, \
             'animacy': 0.069679592172359084, \
             'categorical-recency': 0.046207071175898659}