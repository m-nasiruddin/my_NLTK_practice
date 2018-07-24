from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.corpus import genesis
from itertools import islice
from nltk.corpus.reader.wordnet import information_content as ic
from nltk.stem.wordnet import WordNetLemmatizer as WNLemmma


# words
print(wn.synsets('dog'))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
print(wn.synsets('dog', pos=wn.VERB))
print(wn.synset('dog.n.01'))
print(wn.synset('dog.n.01').definition())
print(len(wn.synset('dog.n.01').examples()))
print(wn.synset('dog.n.01').examples()[0])
print(wn.synset('dog.n.01').lemmas())
print([str(lemma.name()) for lemma in wn.synset('dog.n.01').lemmas()])
print(wn.lemma('dog.n.01.dog').synset())
# download nltk resources/tools
# import nltk
# nltk.download('omw')
print(sorted(wn.langs()))
print(wn.synsets(b'\xe7\x8a\xac'.decode('utf-8'), lang='jpn'))
print(wn.synset('spy.n.01').lemma_names('jpn'))
print(wn.synset('dog.n.01').lemma_names('ita'))
print(wn.lemmas('cane', lang='ita'))
print(sorted(wn.synset('dog.n.01').lemmas('dan')))
print(sorted(wn.synset('dog.n.01').lemmas('por')))
dog_lemma = wn.lemma(b'dog.n.01.c\xc3\xa3o'.decode('utf-8'), lang='por')
print(dog_lemma)
print(dog_lemma.lang())
print(len(wn.all_lemma_names(pos='n', lang='jpn')))
# synsets
dog = wn.synset('dog.n.01')
print(dog.hypernyms())
print(dog.hyponyms())  # doctest: +ELLIPSIS
print(dog.member_holonyms())
print(dog.root_hypernyms())
print(wn.synset('dog.n.01').lowest_common_hypernyms(wn.synset('cat.n.01')))
good = wn.synset('good.a.01')
# print(good.antonyms())  # IT'S NOT POSSIBLE
print(good.lemmas()[0].antonyms())
eat = wn.lemma('eat.v.03.eat')
print(eat)
print(eat.key())
print(eat.count())
print(wn.lemma_from_key(eat.key()))
print(wn.lemma_from_key(eat.key()).synset())
print(wn.lemma_from_key('feebleminded%5:00:00:retarded:00'))
for lemma in wn.synset('eat.v.03').lemmas():
    print(lemma, lemma.count())
for lemma in wn.lemmas('eat', 'v'):
    print(lemma, lemma.count())
vocal = wn.lemma('vocal.a.01.vocal')
print(vocal.derivationally_related_forms())
print(vocal.pertainyms())
print(vocal.antonyms())
# verb frames
print(wn.synset('think.v.01').frame_ids())
for lemma in wn.synset('think.v.01').lemmas():
    print(lemma, lemma.frame_ids())
    print(" | ".join(lemma.frame_strings()))
print(wn.synset('stretch.v.02').frame_ids())
for lemma in wn.synset('stretch.v.02').lemmas():
    print(lemma, lemma.frame_ids())
    print(" | ".join(lemma.frame_strings()))
# similarity
dog = wn.synset('dog.n.01')
cat = wn.synset('cat.n.01')
hit = wn.synset('hit.v.01')
slap = wn.synset('slap.v.01')
print(dog.path_similarity(cat))  # doctest: +ELLIPSIS
print(hit.path_similarity(slap))  # doctest: +ELLIPSIS
print(wn.path_similarity(hit, slap))  # doctest: +ELLIPSIS
print(hit.path_similarity(slap, simulate_root=False))
print(wn.path_similarity(hit, slap, simulate_root=False))
print(dog.lch_similarity(cat))  # doctest: +ELLIPSIS
print(hit.lch_similarity(slap))  # doctest: +ELLIPSIS
print(wn.lch_similarity(hit, slap))  # doctest: +ELLIPSIS
print(hit.lch_similarity(slap, simulate_root=False))
print(wn.lch_similarity(hit, slap, simulate_root=False))
print(dog.wup_similarity(cat))  # doctest: +ELLIPSIS
print(hit.wup_similarity(slap))
print(wn.wup_similarity(hit, slap))
print(hit.wup_similarity(slap, simulate_root=False))
print(wn.wup_similarity(hit, slap, simulate_root=False))
# import nltk
# nltk.download('wordnet_ic')
brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
# import nltk
# nltk.download('genesis')
genesis_ic = wn.ic(genesis, False, 0.0)
print(dog.res_similarity(cat, brown_ic))  # doctest: +ELLIPSIS
print(dog.res_similarity(cat, genesis_ic))  # doctest: +ELLIPSIS
print(dog.jcn_similarity(cat, brown_ic))  # doctest: +ELLIPSIS
print(dog.jcn_similarity(cat, genesis_ic))  # doctest: +ELLIPSIS
print(dog.lin_similarity(cat, semcor_ic))  # doctest: +ELLIPSIS
# access to all synsets
for synset in list(wn.all_synsets('n'))[:10]:
    print(synset)
print(wn.synsets('dog'))  # doctest: +ELLIPSIS
print(wn.synsets('dog', pos='v'))
for synset in islice(wn.all_synsets('n'), 5):
    print(synset, synset.hypernyms())
# morphy
print(wn.morphy('denied', wn.NOUN))
print(wn.morphy('denied', wn.VERB))
print(wn.synsets('denied', wn.NOUN))
print(wn.synsets('denied', wn.VERB))  # doctest: +NORMALIZE_WHITESPACE
print(wn.morphy('dogs'))
print(wn.morphy('churches'))
print(wn.morphy('aardwolves'))
print(wn.morphy('abaci'))
print(wn.morphy('book', wn.NOUN))
print(wn.morphy('hardrock', wn.ADV))
print(wn.morphy('book', wn.ADJ))
print(wn.morphy('his', wn.NOUN))
# synset closures
dog = wn.synset('dog.n.01')
hypo = lambda s: s.hyponyms()
hyper = lambda s: s.hypernyms()
print(list(dog.closure(hypo, depth=1)) == dog.hyponyms())
print(list(dog.closure(hyper, depth=1)) == dog.hypernyms())
print(list(dog.closure(hypo)))
print(list(dog.closure(hyper)))
# regression tests
# morphy returns the base form of a word, if it's input is given as a base form for a POS for which that word is not
# defined
print(wn.synsets('book', wn.NOUN))
print(wn.synsets('book', wn.ADJ))
print(wn.morphy('book', wn.NOUN))
print(wn.morphy('book', wn.ADJ))
# wup_similarity breaks when the two synsets have no common hypernym
t = wn.synsets('picasso')[0]
m = wn.synsets('male')[1]
print(t.wup_similarity(m))  # doctest: +ELLIPSIS
t = wn.synsets('titan')[1]
s = wn.synsets('say', wn.VERB)[0]
print(t.wup_similarity(s))
# "instance of" not included in LCS (very similar to bug 160)
a = wn.synsets("writings")[0]
b = wn.synsets("scripture")[0]
brown_ic = wordnet_ic.ic('ic-brown.dat')
print(a.jcn_similarity(b, brown_ic))  # doctest: +ELLIPSIS
# Verb root IC is zero
s = wn.synsets('say', wn.VERB)[0]
print(ic(s, brown_ic))  # doctest: +ELLIPSIS
# Comparison between WN keys/lemmas should not be case sensitive
k = wn.synsets("jefferson")[0].lemmas()[0].key()
print(wn.lemma_from_key(k))
print(wn.lemma_from_key(k.upper()))
# WordNet root_hypernyms gives incorrect results
for s in wn.all_synsets(wn.NOUN):
    if s.root_hypernyms()[0] != wn.synset('entity.n.01'):
        print(s, s.root_hypernyms())
# JCN Division by zero error
tow = wn.synset('tow.v.01')
shlep = wn.synset('shlep.v.02')
brown_ic = wordnet_ic.ic('ic-brown.dat')
print(tow.jcn_similarity(shlep, brown_ic))  # doctest: +ELLIPSIS
# Depth is zero for instance nouns
s = wn.synset("lincoln.n.01")
print(s.max_depth() > 0)
# Information content smoothing used old reference to all_synsets
genesis_ic = wn.ic(genesis, True, 1.0)
# all_synsets used wrong pos lookup when synsets were cached
for ii in wn.all_synsets():
    pass
for ii in wn.all_synsets():
    pass
# shortest_path_distance ignored instance hypernyms
google = wn.synsets("google")[0]
earth = wn.synsets("earth")[0]
print(google.wup_similarity(earth))  # doctest: +ELLIPSIS
# similarity metrics returned -1 instead of None for no LCS
t = wn.synsets('fly', wn.VERB)[0]
s = wn.synsets('say', wn.VERB)[0]
print(s.shortest_path_distance(t))
print(s.path_similarity(t, simulate_root=False))
print(s.lch_similarity(t, simulate_root=False))
print(s.wup_similarity(t, simulate_root=False))
#  "pants" does not return all the senses it should
print(wn.synsets("pants", 'n'))
# Some nouns not being lemmatised by WordNetLemmatizer().lemmatize
print(WNLemmma().lemmatize("eggs", pos="n"))
print(WNLemmma().lemmatize("legs", pos="n"))
# instance hypernyms not used in similarity calculations
print(wn.synset('john.n.02').lch_similarity(wn.synset('dog.n.01')))  # doctest: +ELLIPSIS
print(wn.synset('john.n.02').wup_similarity(wn.synset('dog.n.01')))  # doctest: +ELLIPSIS
print(wn.synset('john.n.02').res_similarity(wn.synset('dog.n.01'), brown_ic))  # doctest: +ELLIPSIS
print(wn.synset('john.n.02').jcn_similarity(wn.synset('dog.n.01'), brown_ic))  # doctest: +ELLIPSIS
print(wn.synset('john.n.02').lin_similarity(wn.synset('dog.n.01'), brown_ic))  # doctest: +ELLIPSIS
print(wn.synset('john.n.02').hypernym_paths())  # doctest: +ELLIPSIS
# add domains to wordnet
print(wn.synset('code.n.03').topic_domains())
print(wn.synset('pukka.a.01').region_domains())
print(wn.synset('freaky.a.01').usage_domains())
# wordnet failures when python run with -O optimizations
# Run the test suite with python -O to check this
print(wn.synsets("brunch"))
# wordnet returns incorrect result for lowest_common_hypernyms of chef and policeman
print(wn.synset('policeman.n.01').lowest_common_hypernyms(wn.synset('chef.n.01')))
