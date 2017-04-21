__author__ = 'thiagocastroferreira'

from parsers.ReferenceParser import ReferenceParser
from models.baselines import *

from sklearn.metrics import classification_report, accuracy_score

def run():
    references = ReferenceParser(False).run()
    # print "Parse ngrams"
    # bigram, trigram = NgramParser().run()

    y_true = []
    original_y_pred = []
    major_y_pred = []
    participant_y_pred = []
    syntax_y_pred = []
    participant_syntax_y_pred = []
    bigram_y_pred = []
    trigram_y_pred = []

    individual_bigram_y_pred = []
    individual_trigram_y_pred = []

    for reference in references:
        _references = filter(lambda x: x != reference, references)

        y_true.append(utils.type2id[reference['type']])
        original_y_pred.append(original(reference))

        major_y_pred.append(major_slot_choice(reference, _references))

        participant_y_pred.append(major_participant_choice(reference, _references))

        syntax_y_pred.append(major_syntax_choice(reference, _references))

        participant_syntax_y_pred.append(major_choice(['participant-id', 'syncat'], reference, _references))

        bigram_y_pred.append(major_choice(['bigram'], reference, _references))

        trigram_y_pred.append(major_choice(['trigram'], reference, _references))

        try:
            individual_bigram_y_pred.append(major_choice(['bigram', 'participant-id'], reference, _references))
        except:
            individual_bigram_y_pred.append(major_choice(['bigram'], reference, _references))

        try:
            individual_trigram_y_pred.append(major_choice(['trigram', 'participant-id'], reference, _references))
        except:
            individual_trigram_y_pred.append(major_choice(['trigram'], reference, _references))


    print "Baseline Original: "
    print classification_report(y_true, original_y_pred, target_names=['name', 'pronoun', 'description', 'demonstrative', 'empty', 'other'])
    print "Accuracy: ", accuracy_score(y_true, original_y_pred)
    print "\n"
    print "Baseline Slot Most Frequent: "
    print classification_report(y_true, major_y_pred, target_names=['name', 'pronoun', 'description', 'demonstrative', 'empty', 'other'])
    print "Accuracy: ", accuracy_score(y_true, major_y_pred)
    print "\n"
    print "Baseline Participant Most Frequent: "
    print classification_report(y_true, participant_y_pred, target_names=['name', 'pronoun', 'description', 'demonstrative', 'empty', 'other'])
    print "Accuracy: ", accuracy_score(y_true, participant_y_pred)
    print "\n"
    print "Baseline Syntax Most Frequent: "
    print classification_report(y_true, syntax_y_pred, target_names=['name', 'pronoun', 'description', 'demonstrative', 'empty', 'other'])
    print "Accuracy: ", accuracy_score(y_true, syntax_y_pred)
    print "\n"
    print "Baseline Participant+Syntax Most Frequent: "
    print classification_report(y_true, participant_syntax_y_pred, target_names=['name', 'pronoun', 'description', 'demonstrative', 'empty', 'other'])
    print "Accuracy: ", accuracy_score(y_true, participant_syntax_y_pred)
    print "\n"
    print "Baseline Bigram: "
    print classification_report(y_true, bigram_y_pred, target_names=['name', 'pronoun', 'description', 'demonstrative', 'empty', 'other'])
    print "Accuracy: ", accuracy_score(y_true, bigram_y_pred)
    print "\n"
    print "Baseline Trigram: "
    print classification_report(y_true, trigram_y_pred, target_names=['name', 'pronoun', 'description', 'demonstrative', 'empty', 'other'])
    print "Accuracy: ", accuracy_score(y_true, trigram_y_pred)
    print "Baseline Individual Bigram: "
    print classification_report(y_true, individual_bigram_y_pred, target_names=['name', 'pronoun', 'description', 'demonstrative', 'empty', 'other'])
    print "Accuracy: ", accuracy_score(y_true, individual_bigram_y_pred)
    print "\n"
    print "Baseline Individual Trigram: "
    print classification_report(y_true, individual_trigram_y_pred, target_names=['name', 'pronoun', 'description', 'demonstrative', 'empty', 'other'])
    print "Accuracy: ", accuracy_score(y_true, individual_trigram_y_pred)