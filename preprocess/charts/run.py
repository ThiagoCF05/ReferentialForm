__author__ = 'thiagocastroferreira'

import cPickle as p
import os
import numpy as np

from pylab import plot, show, savefig, xlim, figure, \
    hold, ylim, legend, boxplot, setp, axes
from scipy.stats import wilcoxon
import scikits.bootstrap as boot

# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['fliers'][0], color='blue')
    setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    setp(bp['fliers'][2], color='red')
    setp(bp['fliers'][3], color='red')
    setp(bp['medians'][1], color='red')

def run(dir='vareg', features='+syntax+status-recency'):
    print dir
    # Some fake data to plot
    nb_news = p.load(open(os.path.join(dir, 'nb'+features+'+news')))
    mean, sigma = np.mean(nb_news), np.std(nb_news)
    conf_int = boot.ci(nb_news)
    interval = (conf_int[1] - conf_int[0]) / 2
    print 'NB News: '
    print 'Mean:', round(mean, 6)
    print 'STD: ', round(sigma, 6)
    print 'Confidence: ', conf_int
    print 'Interval: ', round(interval, 6)
    print 10 * '-'

    nb_review = p.load(open(os.path.join(dir, 'nb'+features+'+review')))
    mean, sigma = np.mean(nb_review), np.std(nb_review)
    conf_int = boot.ci(nb_review)
    interval = (conf_int[1] - conf_int[0]) / 2
    print 'NB Review: '
    print 'Mean:', round(mean, 6)
    print 'STD: ', round(sigma, 6)
    print 'Confidence: ', conf_int
    print 'Interval: ', round(interval, 6)
    print 10 * '-'

    nb_wiki = p.load(open(os.path.join(dir, 'nb'+features+'+wiki')))
    mean, sigma = np.mean(nb_wiki), np.std(nb_wiki)
    conf_int = boot.ci(nb_wiki)
    interval = (conf_int[1] - conf_int[0]) / 2
    print 'NB Wiki: '
    print 'Mean:', round(mean, 6)
    print 'STD: ', round(sigma, 6)
    print 'Confidence: ', conf_int
    print 'Interval: ', round(interval, 6)
    print 10 * '-'

    rnn_news = p.load(open(os.path.join(dir, 'rnn'+features+'+news')))
    mean, sigma = np.mean(rnn_news), np.std(rnn_news)
    conf_int = boot.ci(rnn_news)
    interval = (conf_int[1] - conf_int[0]) / 2
    print 'RNN News: '
    print 'Mean:', round(mean, 6)
    print 'STD: ', round(sigma, 6)
    print 'Confidence: ', conf_int
    print 'Interval: ', round(interval, 6)
    print 10 * '-'

    rnn_review = p.load(open(os.path.join(dir, 'rnn'+features+'+review')))
    mean, sigma = np.mean(rnn_review), np.std(rnn_review)
    conf_int = boot.ci(rnn_review)
    interval = (conf_int[1] - conf_int[0]) / 2
    print 'RNN Review: '
    print 'Mean:', round(mean, 6)
    print 'STD: ', round(sigma, 6)
    print 'Confidence: ', conf_int
    print 'Interval: ', round(interval, 6)
    print 10 * '-'

    rnn_wiki = p.load(open(os.path.join(dir, 'rnn'+features+'+wiki')))
    mean, sigma = np.mean(rnn_wiki), np.std(rnn_wiki)
    conf_int = boot.ci(rnn_wiki)
    interval = (conf_int[1] - conf_int[0]) / 2
    print 'RNN Wiki: '
    print 'Mean:', round(mean, 6)
    print 'STD: ', round(sigma, 6)
    print 'Confidence: ', conf_int
    print 'Interval: ', round(interval, 6)
    print 10 * '-'

    print 'Wilcoxon NB News X RNN News: ', wilcoxon(nb_news, rnn_news)
    print 'Wilcoxon NB Review X RNN Review: ', wilcoxon(nb_review, rnn_review)
    print 'Wilcoxon NB Wiki X RNN Wiki: ', wilcoxon(nb_wiki, rnn_wiki)

    nb = nb_news
    nb.extend(nb_review)
    nb.extend(nb_wiki)

    rnn = rnn_news
    rnn.extend(rnn_review)
    rnn.extend(rnn_wiki)

    # fig = figure()
    # ax = axes()
    # hold(True)
    #
    # # first boxplot pair
    # bp = boxplot([nb_news, rnn_news], positions = [1, 2], widths = 0.6)
    # setBoxColors(bp)
    #
    # # second boxplot pair
    # bp = boxplot([nb_review, rnn_review], positions = [4, 5], widths = 0.6)
    # setBoxColors(bp)
    #
    # # thrid boxplot pair
    # bp = boxplot([nb_wiki, rnn_wiki], positions = [7, 8], widths = 0.6)
    # setBoxColors(bp)
    #
    # # set axes limits and labels
    # xlim(0,9)
    # ax.set_xticklabels(['News', 'Review', 'Encyclopedic'])
    # ax.set_xticks([1.5, 4.5, 7.5])
    #
    # # draw temporary red and blue lines and use them to create a legend
    # hB, = plot([1,1],'b-')
    # hR, = plot([1,1],'r-')
    # legend((hB, hR),('NB+Syntax+Status-Recency', 'RNN+Syntax+Status-Recency'))
    # hB.set_visible(False)
    # hR.set_visible(False)
    #
    # savefig(dir+'.png')

    return nb, nb_news, nb_review, nb_wiki, rnn, rnn_news, rnn_review, rnn_wiki

nb1, nb1_news, nb1_review, nb1_wiki, rnn1, rnn1_news, rnn1_review, rnn1_wiki = run('vareg')
print '\n'
nb2, nb2_news, nb2_review, nb2_wiki, rnn2, rnn2_news, rnn2_review, rnn2_wiki = run('grec_vareg')

print '\n'
print '+syntax+status-recency'
print 'Wilcoxon NB News: ', wilcoxon(nb1_news, nb2_news)
print 'Wilcoxon NB Review: ', wilcoxon(nb1_review, nb2_review)
print 'Wilcoxon NB Wiki: ', wilcoxon(nb1_wiki, nb2_wiki)
print 'Wilcoxon RNN News: ', wilcoxon(rnn1_news, rnn2_news)
print 'Wilcoxon RNN Review: ', wilcoxon(rnn1_review, rnn2_review)
print 'Wilcoxon RNN Wiki: ', wilcoxon(rnn1_wiki, rnn2_wiki)
# print 'Experiment 1: NB X RNN'
# print 'Wilcoxon: ', wilcoxon(nb1, rnn1)
# print 'Experiment 2: NB X RNN: '
# print 'Wilcoxon: ', wilcoxon(nb2, rnn2)
# print 10 * '-'
# print 'NB1 X NB2'
# print 'Wilcoxon: ', wilcoxon(nb1, nb2)
# print 'RNN1 X RNN2'
# print 'Wilcoxon: ', wilcoxon(rnn1, rnn2)
#
# _nb1, _rnn1 = run('vareg', '+syntax+status+recency')
# _nb2, _rnn2 = run('grec_vareg', '+syntax+status+recency')
# print '\n'
# print '+syntax+status+recency'
# print 'Experiment 1: NB X RNN'
# print 'Wilcoxon: ', wilcoxon(_nb1, _rnn1)
# print 'Experiment 2: NB X RNN'
# print 'Wilcoxon: ', wilcoxon(_nb2, _rnn2)
# print 10 * '-'
# print 'NB1 X NB2'
# print 'Wilcoxon: ', wilcoxon(_nb1, _nb2)
# print 'RNN1 X RNN2'
# print 'Wilcoxon: ', wilcoxon(_rnn1, _rnn2)
# print '\n'
# print 'Experiment 1:'
# print 'nb+syntax+status-recency X nb+syntax+status+recency'
# print 'Wilcoxon: ', wilcoxon(nb1, _nb1)
# print 'rnn+syntax+status-recency X rnn+syntax+status+recency'
# print 'Wilcoxon: ', wilcoxon(rnn1, _rnn1)
# print 10 * '-'
# print 'Experiment 2:'
# print 'nb+syntax+status-recency X nb+syntax+status+recency'
# print 'Wilcoxon: ', wilcoxon(nb2, _nb2)
# print 'rnn+syntax+status-recency X rnn+syntax+status+recency'
# print 'Wilcoxon: ', wilcoxon(rnn2, _rnn2)