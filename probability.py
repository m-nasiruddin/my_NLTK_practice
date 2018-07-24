import nltk
import itertools
import pickle
from nltk.util import unique_list
from nltk.probability import *
from nltk.corpus import brown
from nltk import SimpleGoodTuringProbDist, FreqDist


# freqdist
text1 = ['no', 'good', 'fish', 'goes', 'anywhere', 'without', 'a', 'porpoise', '!']
text2 = ['no', 'good', 'porpoise', 'likes', 'to', 'fish', 'fish', 'anywhere', '.']
fd1 = nltk.FreqDist(text1)
print(fd1 == nltk.FreqDist(text1))
both = nltk.FreqDist(text1 + text2)
both_most_common = both.most_common()
print(list(itertools.chain(*(sorted(ys) for k, ys in itertools.groupby(both_most_common, key=lambda t: t[1])))))
print(both == fd1 + nltk.FreqDist(text2))
print(fd1 == nltk.FreqDist(text1))  # But fd1 is unchanged
fd2 = nltk.FreqDist(text2)
fd1.update(fd2)
print(fd1 == both)
fd2 = nltk.FreqDist(text2)
fd1.update(fd2)
print(fd1 == both)
fd1 = nltk.FreqDist(text1)
fd1.update(text2)
print(fd1 == both)
fd1 = nltk.FreqDist(text1)
fd2 = nltk.FreqDist(fd1)
print(fd2 == fd1)
fd1 = nltk.FreqDist(text1)
pickled = pickle.dumps(fd1)
print(fd1 == pickle.loads(pickled))
# testing some hmm estimators
corpus = nltk.corpus.brown.tagged_sents(categories='adventure')[:500]
print(len(corpus))
tag_set = unique_list(tag for sent in corpus for (word, tag) in sent)
print(len(tag_set))
symbols = unique_list(word for sent in corpus for (word, tag) in sent)
print(len(symbols))
print(len(tag_set))
symbols = unique_list(word for sent in corpus for (word, tag) in sent)
print(len(symbols))
trainer = nltk.tag.HiddenMarkovModelTrainer(tag_set, symbols)
train_corpus = []
test_corpus = []
for i in range(len(corpus)):
    if i % 10:
        train_corpus += [corpus[i]]
    else:
        test_corpus += [corpus[i]]
print(len(train_corpus))
print(len(test_corpus))


def train_and_test(est):
    hmm = trainer.train_supervised(train_corpus, estimator=est)
    print('%.2f%%' % (100 * hmm.evaluate(test_corpus)))


# maximum likelihood estimation
mle = lambda fd, bins: MLEProbDist(fd)
print(train_and_test(mle))
print(train_and_test(LaplaceProbDist))
print(train_and_test(ELEProbDist))
def lidstone(gamma):
    return lambda fd, bins: LidstoneProbDist(fd, gamma, bins)
print(train_and_test(lidstone(0.1)))
print(train_and_test(lidstone(0.5)))
print(train_and_test(lidstone(1.0)))
# witten bell estimation
print(train_and_test(WittenBellProbDist))
gt = lambda fd, bins: SimpleGoodTuringProbDist(fd, bins=1e5)
print(train_and_test(gt))
# kneser ney estimation
corpus = [[((x[0],y[0],z[0]),(x[1],y[1],z[1]))
    for x, y, z in nltk.trigrams(sent)]
        for sent in corpus[:100]]
tag_set = unique_list(tag for sent in corpus for (word,tag) in sent)
print(len(tag_set))
symbols = unique_list(word for sent in corpus for (word,tag) in sent)
print(len(symbols))
trainer = nltk.tag.HiddenMarkovModelTrainer(tag_set, symbols)
train_corpus = []
test_corpus = []
for i in range(len(corpus)):
    if i % 10:
        train_corpus += [corpus[i]]
    else:
        test_corpus += [corpus[i]]
print(len(train_corpus))
print(len(test_corpus))
kn = lambda fd, bins: KneserNeyProbDist(fd)
print(train_and_test(kn))
# squashed bugs
fd = nltk.FreqDist('a')
print(list(fd.keys()))
print(fd.pop('a'))
print(list(fd.keys()))
fd = nltk.FreqDist('aab')
print(list(fd._cumulative_frequencies(['a'])))
print(list(fd._cumulative_frequencies(['a', 'b'])))
fd = FreqDist('aab')
fd.clear()
print(fd.N())
# print(brown.fileids('blah'))
print(brown.categories())
fd = FreqDist({'a':1, 'b':1, 'c': 2, 'd': 3, 'e': 4, 'f': 4, 'g': 4, 'h': 5, 'i': 5, 'j': 6, 'k': 6, 'l': 6, 'm': 7, 'n': 7, 'o': 8, 'p': 9, 'q': 10})
p = SimpleGoodTuringProbDist(fd)
print(p.prob('a'))
print(p.prob('o'))
print(p.prob('z'))
print(p.prob('foobar'))
pd = MLEProbDist(fd)
sorted(pd.samples()) == sorted(pickle.loads(pickle.dumps(pd)).samples())
dpd = DictionaryConditionalProbDist({'x': pd})
unpickled = pickle.loads(pickle.dumps(dpd))
print(dpd['x'].prob('a'))
dpd['x'].prob('a') == unpickled['x'].prob('a')
cfd = nltk.probability.ConditionalFreqDist()
cfd['foo']['hello'] += 1
cfd['foo']['hello'] += 1
cfd['bar']['hello'] += 1
cfd2 = pickle.loads(pickle.dumps(cfd))
print(cfd2 == cfd)
cpd = ConditionalProbDist(cfd, SimpleGoodTuringProbDist)
cpd2 = pickle.loads(pickle.dumps(cpd))
print(cpd['foo'].prob('hello') == cpd2['foo'].prob('hello'))
