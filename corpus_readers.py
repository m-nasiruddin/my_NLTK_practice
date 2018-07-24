import nltk
import nltk.corpus, os.path, textwrap
from nltk.corpus import inaugural, brown, indian, conll2000, conll2002, semcor, ieer, treebank, ptb, sinica_treebank,\
    conll2007, names, stopwords, words, cmudict, movie_reviews, reuters, comparative_sentences, opinion_lexicon,\
    ppattach, product_reviews_1, pros_cons, senseval, sentence_polarity, shakespeare, subjectivity, timit,\
    twitter_samples, rte, verbnet, abc, genesis, state_union, webtext, qc
from xml.etree import ElementTree as ET
from nltk.corpus.reader.api import *
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.tagged import TaggedCorpusReader
from nltk.corpus.reader.util import *
from io import StringIO, BytesIO
from nltk.data import SeekableUnicodeStreamReader


# tagged corpus reader
# overview
# The Brown corpus:
print(str(nltk.corpus.brown).replace('\\\\', '/'))
# The Penn Treebank Corpus:
print(str(nltk.corpus.treebank).replace('\\\\', '/'))
# The Name Genders Corpus:
print(str(nltk.corpus.names).replace('\\\\', '/'))
# The Inaugural Address Corpus:
# nltk.download('inaugural')
print(str(nltk.corpus.inaugural).replace('\\\\', '/'))
print(nltk.corpus.treebank.fileids())  # doctest: +ELLIPSIS
print(nltk.corpus.inaugural.fileids())  # doctest: +ELLIPSIS
print(inaugural.raw('1789-Washington.txt'))  # doctest: +ELLIPSIS
print(inaugural.words('1789-Washington.txt'))
print(inaugural.sents('1789-Washington.txt'))  # doctest: +ELLIPSIS
print(inaugural.paras('1789-Washington.txt'))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
l1 = len(inaugural.words('1789-Washington.txt'))
l2 = len(inaugural.words('1793-Washington.txt'))
l3 = len(inaugural.words(['1789-Washington.txt', '1793-Washington.txt']))
print('%s+%s == %s' % (l1, l2, l3))
print(len(inaugural.words()))
print(inaugural.readme()[:32])
# plain text corpora
# nltk.download('abc')
print(nltk.corpus.abc.words())
print(nltk.corpus.genesis.words())
# nltk.download('gutenberg')
print(nltk.corpus.gutenberg.words(fileids='austen-emma.txt'))
print(nltk.corpus.inaugural.words())
# nltk.download('state_union')
print(nltk.corpus.state_union.words())
# nltk.download('webtext')
print(nltk.corpus.webtext.words())
# tagged corpora
print(brown.words())
print(brown.tagged_words())
print(brown.sents())  # doctest: +ELLIPSIS
print(brown.tagged_sents())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
print(brown.paras(categories='reviews'))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
print(brown.tagged_paras(categories='reviews'))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
# nltk.download('indian')
print(indian.words())  # doctest: +SKIP
print(indian.tagged_words())  # doctest: +SKIP
# nltk.download('universal_tagset')
print(brown.tagged_sents(tagset='universal'))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
print(conll2000.tagged_words(tagset='universal'))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
# chunked corpora
print(conll2000.sents())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
for tree in conll2000.chunked_sents()[:2]:
    print(tree)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
# nltk.download('conll2002')
print(conll2002.sents())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
for tree in conll2002.chunked_sents()[:2]:
    print(tree)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
# nltk.download('semcor')
print(semcor.words())
print(semcor.chunks())
print(semcor.sents())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
print(semcor.chunk_sents())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
print(list(map(str, semcor.tagged_chunks(tag='both')[:3])))
print([[str(c) for c in s] for s in semcor.tagged_sents(tag='both')[:2]])
# nltk.download('ieer')
print(ieer.fileids())  # doctest: +NORMALIZE_WHITESPACE
docs = ieer.parsed_docs('APW_19980314')
print(docs[0])
print(docs[0].docno)
print(docs[0].doctype)
print(docs[0].date_time)
print(docs[0].headline)
print(docs[0].text)  # doctest: +ELLIPSIS
# parsed corpora
print(treebank.fileids())  # doctest: +ELLIPSIS
print(treebank.words('wsj_0003.mrg'))
print(treebank.tagged_words('wsj_0003.mrg'))
print(treebank.parsed_sents('wsj_0003.mrg')[0])  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
# nltk.download('ptb')
print(ptb.fileids())  # doctest: +SKIP
# download the corpus from here: https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/treebank.zip
# then extract and place to the following location: .../nltk_data/corpora/ptb/
print(ptb.words('treebank/combined/wsj_0003.mrg'))  # doctest: +SKIP
print(ptb.tagged_words('treebank/combined/wsj_0003.mrg'))  # doctest: +SKIP
# print(ptb.categories())  # doctest: +SKIP
# print(ptb.fileids('news'))  # doctest: +SKIP
# print(ptb.words(categories=['humor', 'fiction']))  # doctest: +SKIP
# nltk.download('sinica_treebank')
print(sinica_treebank.sents())  # doctest: +SKIP
print(sinica_treebank.parsed_sents()[25])  # doctest: +SKIP
# nltk.download('conll2007')
print(conll2007.sents('esp.train')[0])  # doctest: +SKIP
print(conll2007.parsed_sents('esp.train')[0])  # doctest: +SKIP
print(conll2007.parsed_sents('esp.train')[0].tree())  # doctest: +SKIP
# for tree in ycoe.parsed_sents('cocuraC')[:4]:
#     print(tree)  # doctest: +SKIP
# word lists and lexicons
print(words.fileids())
print(words.words('en'))  # doctest: +ELLIPSIS
print(stopwords.fileids())  # doctest: +ELLIPSIS
print(stopwords.words('portuguese'))  # doctest: +ELLIPSIS
# nltk.download('names')
print(names.fileids())
print(names.words('male.txt'))  # doctest: +ELLIPSIS
print(names.words('female.txt'))  # doctest: +ELLIPSIS
# nltk.download('cmudict')
print(cmudict.entries()[653:659])  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
# Load the entire cmudict corpus into a Python dictionary:
transcr = cmudict.dict()
print([transcr[w][0] for w in 'Natural Language Tool Kit'.lower().split()])  # doctest: +NORMALIZE_WHITESPACE
# categorized corpora
print(brown.categories())  # doctest: +NORMALIZE_WHITESPACE
print(movie_reviews.categories())
# nltk.download('reuters')
print(reuters.categories())  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
print(brown.categories('ca01'))
print(brown.categories(['ca01', 'cb01']))
print(reuters.categories('training/9865'))
print(reuters.categories(['training/9865', 'training/9880']))
print(reuters.fileids('barley'))  # doctest: +ELLIPSIS
print(brown.tagged_words(categories='news'))
print(brown.sents(categories=['editorial', 'reviews']))  # doctest: +NORMALIZE_WHITESPACE


def classify(doc1):
    return 'news'  # Trivial classifier


doc1 = 'ca01'
print(classify(doc1) in brown.categories(doc1))
# other corpora
# comparative_sentences
# nltk.download('comparative_sentences')
comparison = comparative_sentences.comparisons()[0]
print(comparison.text)
print(comparison.entity_2)
print(comparison.feature, comparison.keyword)
print(len(comparative_sentences.comparisons()))
# opinion_lexicon
# nltk.download('opinion_lexicon')
print(opinion_lexicon.words()[:4])
print(opinion_lexicon.negative()[:4])
print(opinion_lexicon.words()[0:10])
print(sorted(opinion_lexicon.words())[0:10])
# ppattach
# nltk.download('ppattach')
print(ppattach.attachments('training'))  # doctest: +NORMALIZE_WHITESPACE
inst = ppattach.attachments('training')[0]
print(inst.sent, inst.verb, inst.noun1, inst.prep, inst.noun2)
print(inst.attachment)
# product_reviews_1 and product_reviews_2
# nltk.download('product_reviews_1')
camera_reviews = product_reviews_1.reviews('Canon_G3.txt')
review = camera_reviews[0]
print(review.sents()[0])
print(review.features())
print(product_reviews_1.features('Canon_G3.txt'))
n_reviews = len([(feat, score) for (feat, score) in product_reviews_1.features('Canon_G3.txt') if feat == 'picture'])
tot = sum([int(score) for (feat, score) in product_reviews_1.features('Canon_G3.txt') if feat == 'picture'])
# We use float for backward compatibility with division in Python2.7
mean = float(tot) / n_reviews
print(n_reviews, tot, mean)
# pros_cons
# nltk.download('pros_cons')
print(pros_cons.sents(categories='Cons'))
print(pros_cons.words('IntegratedPros.txt'))
# semcor
print(semcor.words('brown2/tagfiles/br-n12.xml'))  # doctest: +ELLIPSIS
sent = semcor.xml('brown2/tagfiles/br-n12.xml').findall('context/p/s')[0]
for wordform in sent.getchildren():
    print(wordform.text)
    for key in sorted(wordform.keys()):
        print(key + '=' + wordform.get(key))
    print()
# senseval
# nltk.download('senseval')
print(senseval.fileids())
print(senseval.instances('hard.pos'))
for inst in senseval.instances('interest.pos')[:10]:
    p = inst.position
    left = ' '.join(w for (w, t) in inst.context[p - 2:p])
    word = ' '.join(w for (w, t) in inst.context[p:p + 1])
    right = ' '.join(w for (w, t) in inst.context[p + 1:p + 3])
    senses = ' '.join(inst.senses)
    print('%20s |%10s | %-15s -> %s' % (left, word, right, senses))
# sentence_polarity
# nltk.download('sentence_polarity')
print(sentence_polarity.sents())
print(sentence_polarity.categories())
print(sentence_polarity.sents()[1])
# shakespeare
# nltk.download('shakespeare')
print(shakespeare.fileids())  # doctest: +ELLIPSIS
play = shakespeare.xml('dream.xml')
print(play)  # doctest: +ELLIPSIS
print('%s: %s' % (play[0].tag, play[0].text))
personae = [persona.text for persona in play.findall('PERSONAE/PERSONA')]
print(personae)  # doctest: +ELLIPSIS
# Find and print speakers not listed as personae
names = [persona.split(',')[0] for persona in personae]
speakers = set(speaker.text for speaker in play.findall('*/*/*/SPEAKER'))
print(sorted(speakers.difference(names)))  # doctest: +NORMALIZE_WHITESPACE
# subjectivity
# nltk.download('subjectivity')
print(subjectivity.categories())
print(subjectivity.sents()[23])
print(subjectivity.words(categories='subj'))
# timit
# nltk.download('timit')
print(timit.utteranceids())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
item = timit.utteranceids()[5]
print(timit.phones(item))  # doctest: +NORMALIZE_WHITESPACE
print(timit.words(item))
print(timit.play(item))  # doctest: +SKIP
for tree in timit.phone_trees(item):
    print(tree)
print(timit.phone_times(item))  # doctest: +ELLIPSIS
print(timit.word_times(item))  # doctest: +ELLIPSIS
print(timit.sent_times(item))
print(timit.play(item, 2190, 8804))  # 'clasp'  # doctest: +SKIP
print(timit.spkrid(item))
print(timit.sentid(item))
print(timit.spkrinfo(timit.spkrid(item)))  # doctest: +NORMALIZE_WHITESPACE
#  List the speech samples from the same speaker:
print(timit.utteranceids(spkrid=timit.spkrid(item)))  # doctest: +ELLIPSIS
# twitter_samples
# nltk.download('twitter_samples')
print(twitter_samples.fileids())
print(twitter_samples.strings('tweets.20150430-223406.json'))
print(twitter_samples.tokenized('tweets.20150430-223406.json'))
# rte
# nltk.download('rte')
print(rte.fileids())  # doctest: +ELLIPSIS
rtepairs = rte.pairs(['rte2_test.xml', 'rte3_test.xml'])
print(rtepairs)  # doctest: +ELLIPSIS
print(rtepairs[5])
print(rtepairs[5].text)  # doctest: +NORMALIZE_WHITESPACE
print(rtepairs[5].hyp)
print(rtepairs[5].value)
xmltree = rte.xml('rte3_dev.xml')
print(xmltree)  # doctest: +SKIP
print(xmltree[7].findtext('t'))  # doctest: +NORMALIZE_WHITESPACE
# verbnet
# nltk.download('verbnet')
print(verbnet.lemmas()[20:25])
print(verbnet.classids()[:5])
print(verbnet.classids('accept'))
print(verbnet.vnclass('remove-10.1'))  # doctest: +ELLIPSIS
print(verbnet.vnclass('10.1'))  # doctest: +ELLIPSIS
vn_31_2 = verbnet.vnclass('admire-31.2')
for themrole in vn_31_2.findall('THEMROLES/THEMROLE'):
    print(themrole.attrib['type'])
    for selrestr in themrole.findall('SELRESTRS/SELRESTR'):
        print('[%(Value)s%(type)s]' % selrestr.attrib)
    print()
print(verbnet.pprint('57'))
# nps_chat
# nltk.download('nps_chat')
print(nltk.corpus.nps_chat.words())
print(nltk.corpus.nps_chat.tagged_words())
print(nltk.corpus.nps_chat.tagged_posts())  # doctest: +NORMALIZE_WHITESPACE
print(nltk.corpus.nps_chat.xml_posts())  # doctest: +ELLIPSIS
posts = nltk.corpus.nps_chat.xml_posts()
print(sorted(nltk.FreqDist(p.attrib['class'] for p in posts).keys()))
print(posts[0].text)
tokens = posts[0].findall('terminals/t')
print([t.attrib['pos'] + "/" + t.attrib['word'] for t in tokens])
# multext_east
# nltk.download('mte_teip5')
print(nltk.corpus.multext_east.words("oana-en.xml"))
print(nltk.corpus.multext_east.tagged_words("oana-en.xml"))
print(nltk.corpus.multext_east.tagged_sents("oana-en.xml", "universal"))
# corpus reader classes
# automatically created corpus reader instances
print(nltk.corpus.brown)  # doctest: +ELLIPSIS
print(nltk.corpus.treebank)  # doctest: +ELLIPSIS
print(nltk.corpus.names)  # doctest: +ELLIPSIS
print(nltk.corpus.genesis)  # doctest: +ELLIPSIS
print(nltk.corpus.inaugural)  # doctest: +ELLIPSIS
# creating new corpus reader instances
# Find the directory where the corpus lives.
genesis_dir = nltk.data.find('corpora/genesis')
# Create our custom sentence tokenizer.
my_sent_tokenizer = nltk.RegexpTokenizer('[^.!?]+')
# Create the new corpus reader object.
my_genesis = nltk.corpus.PlaintextCorpusReader(genesis_dir, '.*\.txt', sent_tokenizer=my_sent_tokenizer)
# Use the new corpus reader object.
print(my_genesis.sents('english-kjv.txt')[0])  # doctest: +NORMALIZE_WHITESPACE
my_corpus = nltk.corpus.PlaintextCorpusReader(
    '/home/osboxes/Nextcloud/PhD/workspaces/PycharmProjects/algorithms/nltk/data/input', '.*\.txt')  # doctest: +SKIP
# common corpus reader methods
print(str(nltk.corpus.genesis.root).replace(os.path.sep, '/'))  # doctest: +ELLIPSIS
some_corpus_file_id = nltk.corpus.reuters.fileids()[0]
print(os.path.normpath(some_corpus_file_id).replace(os.path.sep, '/'))
print(nltk.corpus.timit.fileids())  # doctest: +ELLIPSIS
print(nltk.corpus.timit.fileids('phn'))  # doctest: +ELLIPSIS
print(nltk.corpus.brown.fileids('hobbies'))  # doctest: +ELLIPSIS
print(str(nltk.corpus.brown.abspath('ce06')).replace(os.path.sep, '/'))  # doctest: +ELLIPSIS
# data access methods
print(nltk.corpus.brown.words())  # list of str
print(nltk.corpus.brown.sents())  # list of (list of str)
print(nltk.corpus.brown.paras())  # list of (list of (list of str))
print(nltk.corpus.brown.tagged_words())  # list of (str,str) tuple
print(nltk.corpus.brown.tagged_sents())  # list of (list of (str,str))
print(nltk.corpus.brown.tagged_paras())  # list of (list of (list of (str,str)))
# print(nltk.corpus.brown.chunked_sents())  # list of (Tree w/ (str,str) leaves)
# print(nltk.corpus.brown.parsed_sents())  # list of (Tree with str leaves)
# print(nltk.corpus.brown.parsed_paras())  # list of (list of (Tree with str leaves))
# print(nltk.corpus.brown.xml())  # A single xml ElementTree
print(nltk.corpus.brown.raw())  # str (unprocessed corpus contents)
print(nltk.corpus.brown.words())
print(nltk.corpus.treebank.words())
print(nltk.corpus.conll2002.words())
print(nltk.corpus.genesis.words())
print(nltk.corpus.brown.tagged_words())
print(nltk.corpus.treebank.tagged_words())
print(nltk.corpus.conll2002.tagged_words())
# print(nltk.corpus.genesis.tagged_words())
print(nltk.corpus.timit.utteranceids())  # doctest: +ELLIPSIS
print(nltk.corpus.timit.words('dr1-fvmh0/sa2'))
print(nltk.corpus.timit.fileids())  # doctest: +ELLIPSIS
# print(nltk.corpus.timit.words('dr1-fvmh0/sa1.txt'))  # doctest: +SKIP
# nltk.download('propbank')
roleset = nltk.corpus.propbank.roleset('eat.01')
print(ET.tostring(roleset).decode('utf8'))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE


# writing new corpus readers
# implementation


# constructor
def __init__(self, root, files, chunk_types):
    CorpusReader.__init__(self, root, files)
    self.chunk_types = tuple(chunk_types)


# data access methods
print(str(nltk.corpus.brown.abspaths()).replace('\\\\', '/'))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
print(str(nltk.corpus.brown.abspaths('ce06')).replace('\\\\', '/'))  # doctest: +ELLIPSIS
print(str(nltk.corpus.brown.abspaths(['ce06', 'ce07'])).replace('\\\\', '/'))  # doctest: +ELLIPSIS


# +NORMALIZE_WHITESPACE


def words(self, fileids=None):
    return concat([self.CorpusView(fileid, self._read_word_block) for fileid in self.abspaths(fileids)])


def words(self, fileids=None):
    return concat([[w1 for w1 in open(fileid).read().split('\n') if w1] for fileid in self.abspaths(fileids)])


# corpus view
def simple_block_reader(stream):
    return stream.readline().split()


# regression tests
def make_testcorpus(ext='', **fileids):
    root = tempfile.mkdtemp()
    for fileid, contents in fileids.items():
        fileid += ext
        f = open(os.path.join(root, fileid), 'w')
        f.write(textwrap.dedent(contents))
        f.close()
    return root


def del_testcorpus(root):
    for fileid in os.listdir(root):
        os.remove(os.path.join(root, fileid))
        os.rmdir(root)


# plaintext corpus reader
root = make_testcorpus(ext='.txt',
                       a="""
                       This is the first sentence.  Here is another
                       sentence!  And here's a third sentence.

                       This is the second paragraph.  Tokenization is currently
                       fairly simple, so the period in Mr. gets tokenized.
                       """,
                       b="""This is the second file.""")
corpus = PlaintextCorpusReader(root, ['a.txt', 'b.txt'])
print(corpus.fileids())
corpus = PlaintextCorpusReader(root, '.*\.txt')
print(corpus.fileids())
print(str(corpus.root) == str(root))
print(corpus.words())
print(corpus.raw()[:40])
print(len(corpus.words()), [len(corpus.words(d)) for d in corpus.fileids()])
print(corpus.words('a.txt'))
print(corpus.words('b.txt'))
print(corpus.words()[:4], corpus.words()[-4:])
# del_testcorpus(root)
for corpus in (abc, genesis, inaugural, state_union, webtext):
    print(str(corpus).replace('\\\\', '/'))
    print('  ', repr(corpus.fileids())[:60])
    print('  ', repr(corpus.words()[:10])[:60])
root = make_testcorpus(
    a="""
    This/det is/verb the/det first/adj sentence/noun ./punc
    Here/det  is/verb  another/adj    sentence/noun ./punc
    Note/verb that/comp you/pron can/verb use/verb
    any/noun tag/noun set/noun

    This/det is/verb the/det second/adj paragraph/noun ./punc
    word/n without/adj a/det tag/noun :/: hello ./punc
    """,
    b="""
    This/det is/verb the/det second/adj file/noun ./punc
    """)
corpus = TaggedCorpusReader(root, list('ab'))
print(corpus.fileids())
print(str(corpus.root) == str(root))
print(corpus.words())
print(corpus.sents())  # doctest: +ELLIPSIS
print(corpus.paras())  # doctest: +ELLIPSIS
print(corpus.tagged_words())  # doctest: +ELLIPSIS
print(corpus.tagged_sents())  # doctest: +ELLIPSIS
print(corpus.tagged_paras())  # doctest: +ELLIPSIS
print(corpus.raw()[:40])
print(len(corpus.words()), [len(corpus.words(d)) for d in corpus.fileids()])
print(len(corpus.sents()), [len(corpus.sents(d)) for d in corpus.fileids()])
print(len(corpus.paras()), [len(corpus.paras(d)) for d in corpus.fileids()])
print(corpus.words('a'))
print(corpus.words('b'))
# del_testcorpus(root)
print(brown.fileids())  # doctest: +ELLIPSIS
print(brown.categories())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
print(repr(brown.root).replace('\\\\', '/'))  # doctest: +ELLIPSIS
print(brown.words())
print(brown.sents())  # doctest: +ELLIPSIS
print(brown.paras())  # doctest: +ELLIPSIS
print(brown.tagged_words())  # doctest: +ELLIPSIS
print(brown.tagged_sents())  # doctest: +ELLIPSIS
print(brown.tagged_paras())  # doctest: +ELLIPSIS
# verbnet corpus reader
print(len(verbnet.lemmas()))
print(len(verbnet.wordnetids()))
print(len(verbnet.classids()))
print(verbnet.classids(lemma='take'))  # doctest: +NORMALIZE_WHITESPACE
print(verbnet.classids(wordnetid='lead%2:38:01'))
print(verbnet.classids(fileid='approve-77.xml'))
print(verbnet.classids(classid='admire-31.2'))  # subclasses
a = ElementTree.tostring(verbnet.vnclass('admire-31.2.xml'))
b = ElementTree.tostring(verbnet.vnclass('admire-31.2'))
c = ElementTree.tostring(verbnet.vnclass('31.2'))
print(a == b == c)
print(verbnet.fileids('admire-31.2'))
print(verbnet.fileids(['admire-31.2', 'obtain-13.5.2']))
# print(verbnet.fileids('badidentifier'))
print(verbnet.longid('31.2'))
print(verbnet.longid('admire-31.2'))
print(verbnet.shortid('31.2'))
print(verbnet.shortid('admire-31.2'))
# print(verbnet.longid('badidentifier'))
# print(verbnet.shortid('badidentifier'))
# corpus view regression tests
# A very short file (160 chars):
f1 = nltk.data.find('corpora/inaugural/README')
# A relatively short file (791 chars):
f2 = nltk.data.find('corpora/inaugural/1793-Washington.txt')
# A longer file (32k chars):
f3 = nltk.data.find('corpora/inaugural/1909-Taft.txt')
fileids = [f1, f2, f3]
# concatenation
c1 = StreamBackedCorpusView(f1, read_whitespace_block, encoding='utf-8')
c2 = StreamBackedCorpusView(f2, read_whitespace_block, encoding='utf-8')
c3 = StreamBackedCorpusView(f3, read_whitespace_block, encoding='utf-8')
c123 = c1+c2+c3
print(c123)
l1 = f1.open(encoding='utf-8').read().split()
l2 = f2.open(encoding='utf-8').read().split()
l3 = f3.open(encoding='utf-8').read().split()
l123 = l1+l2+l3
print(list(c123) == l123)
print((c1+c2+c3)[100] == l123[100])
# slicing
c1 = StreamBackedCorpusView(f1, read_whitespace_block, encoding='utf-8')
l1 = f1.open(encoding='utf-8').read().split()
print(len(c1))
print(len(c1) < LazySubsequence.MIN_SIZE)
indices = [-60, -30, -22, -21, -20, -1, 0, 1, 10, 20, 21, 22, 30, 60]
for s in indices:
    for e in indices:
        assert list(c1[s:e]) == l1[s:e]
for s in indices:
    assert list(c1[s:]) == l1[s:]
for e in indices:
    assert list(c1[:e]) == l1[:e]
print(list(c1[:]) == list(l1[:]))
c3 = StreamBackedCorpusView(f3, read_whitespace_block, encoding='utf-8')
l3 = f3.open(encoding='utf-8').read().split()
print(len(c3))
print(len(c3) > LazySubsequence.MIN_SIZE*2)
indices = [-12000, -6000, -5431, -5430, -5429, -3000, -200, -1, 0, 1, 200, 3000, 5000, 5429, 5430, 5431, 6000, 12000]
for s in indices:
    for e in indices:
        assert list(c3[s:e]) == l3[s:e]
for s in indices:
    assert list(c3[s:]) == l3[s:]
for e in indices:
    assert list(c3[:e]) == l3[:e]
print(list(c3[:]) == list(l3[:]))
c3 = StreamBackedCorpusView(f3, read_whitespace_block)
iterators = [c3.iterate_from(n) for n in [0, 15, 30, 45]]
for i in range(15):
    for iterator in iterators:
        print('%-15s' % next(iterator))
    print()
# SeekableUnicodeStreamReader
stream = BytesIO(b"""
This is a test file.
It is encoded in ascii.
""".decode('ascii').encode('ascii'))
reader = SeekableUnicodeStreamReader(stream, 'ascii')
print(reader.read())  # read the entire file.
print(reader.seek(0))  # rewind to the start.
print(reader.read(5))  # read at most 5 bytes.
print(reader.readline())  # read to the end of the line.
print(reader.seek(0))  # rewind to the start.
for line in reader:
    print(repr(line))  # iterate over lines
print(reader.seek(0))  # rewind to the start.
print(reader.readlines())  # read a list of line strings
print(reader.close())
# size argument to read()
stream = BytesIO(b"""
This is a test file.
It is encoded in utf-16.
""".decode('ascii').encode('utf-16'))
reader = SeekableUnicodeStreamReader(stream, 'utf-16')
print(reader.read(10))
print(reader.seek(0))  # rewind to the start.
print(reader.read(1))  # we actually need to read 4 bytes
print(int(reader.tell()))
print(reader.seek(0))  # rewind to the start.
print(reader.readline())  # stores extra text in a buffer
print(reader.linebuffer)  # examine the buffer contents
print(reader.read(0))  # returns the contents of the buffer
print(reader.linebuffer)  # examine the buffer contents
# seek ans tell
stream = BytesIO(b"""
This is a test file.
It is encoded in utf-16.
""".decode('ascii').encode('utf-16'))
reader = SeekableUnicodeStreamReader(stream, 'utf-16')
print(reader.read(20))
pos = reader.tell()
print(pos)
print(reader.read(20))
print(reader.seek(pos))  # rewind to the position from tell.
print(reader.read(20))
stream = BytesIO(b"""
This is a test file.
It is encoded in utf-16.
""".decode('ascii').encode('utf-16'))
reader = SeekableUnicodeStreamReader(stream, 'utf-16')
print(reader.readline())
pos = reader.tell()
print(pos)
print(reader.readline())
print(reader.seek(pos))  # rewind to the position from tell.
print(reader.readline())
# squashed bugs
f = StringIO(b"""
(a b c)
# This line is a comment.
(d e f\ng h)""".decode('ascii'))
print(read_sexpr_block(f, block_size=38, comment_char='#'))
print(read_sexpr_block(f, block_size=38, comment_char='#'))
f = StringIO(b"""
This file ends mid-sexpr
(hello (world""".decode('ascii'))
for i in range(3):
    print(read_sexpr_block(f))
f = StringIO(b"This file has no trailing whitespace.".decode('ascii'))
for i in range(3):
    print(read_sexpr_block(f))
# Bug fixed in 5279:
f = StringIO(b"a b c)".decode('ascii'))
for i in range(3):
    print(read_sexpr_block(f))
sents = nltk.corpus.brown.sents()
print(sents[6000])
print(sents[6000])
print(reuters.words('training/13085'))
print(reuters.words('training/5082'))
nltk.download('qc')
print(qc.tuples('test.txt'))
