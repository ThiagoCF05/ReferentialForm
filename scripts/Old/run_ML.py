__author__ = 'thiagocastroferreira'

from parsers.ReferenceParser import ReferenceParser
import utils
import numpy as np

from sklearn.cross_validation import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

def run():
    parser = ReferenceParser(False)
    references = parser.run()
    references = parser.parse_frequencies(references)
    del parser

    X = np.array(map(lambda x: [#int(x['list-id'][-1]), \
                       int(x['paragraph-id']), \
                       int(x['reference-id']), \
                       int(x['recency']), \
                       int(x['position']), \
                       # float(x['entropy']), \
                       int(utils.syntax2id[x['syncat']]), \
                       int(utils.topic2id[x['topic']]), \
                       int(utils.topic2id[x['competitor']]), \
                       int(x['participant-id']), \
                       int(x['freq-name']), \
                       int(x['freq-pronoun']), \
                       int(x['freq-description']), \
                       int(x['freq-demonstrative']), \
                       int(x['freq-empty']) \
                       ], references))
    y = np.array(map(lambda x: utils.type2id[x['type']], references))

    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)

    clf = DecisionTreeClassifier(random_state=0)
    predicts = cross_val_predict(clf, X, y, cv=10)

    print "Decision Tree: "
    print classification_report(y, predicts, target_names=['name', 'pronoun', 'description', 'demonstrative', 'empty', 'other'])
    print "Accuracy: ", accuracy_score(y, predicts)
    print "\n"

    clf = RandomForestClassifier(n_estimators=100)
    predicts = cross_val_predict(clf, X, y, cv=10)

    print "Random Forest: "
    print classification_report(y, predicts, target_names=['name', 'pronoun', 'description', 'demonstrative', 'empty', 'other'])
    print "Accuracy: ", accuracy_score(y, predicts)
    print "\n"

    clf = svm.SVC()
    predicts = cross_val_predict(clf, X, y, cv=10)

    print "SVM: "
    print classification_report(y, predicts, target_names=['name', 'pronoun', 'description', 'demonstrative', 'empty', 'other'])
    print "Accuracy: ", accuracy_score(y, predicts)
    print "\n"

run()