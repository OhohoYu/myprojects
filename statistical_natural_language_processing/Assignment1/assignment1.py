#! SETUP 1 (RIEDEL)
import sys, os
_snlp_book_dir = "../../../../"
sys.path.append(_snlp_book_dir)
import statnlpbook.lm as lm
import statnlpbook.ohhla as ohhla
import math

#! SETUP 2 (RIEDEL)
_snlp_train_dir = _snlp_book_dir + "/data/ohhla/train"
_snlp_dev_dir = _snlp_book_dir + "/data/ohhla/dev"
_snlp_train_song_words = ohhla.words(ohhla.load_all_songs(_snlp_train_dir))
_snlp_dev_song_words = ohhla.words(ohhla.load_all_songs(_snlp_dev_dir))
assert(len(_snlp_train_song_words)==1041496)

from collections import *


class KneserNeyBigram(lm.LanguageModel):
    """
    This class applies interpolated Kneser-Ney smoothing, an extension
    of absolute discounting with a clever way of constructing the
    lower-order (backoff) model. More detailed explanations are provided
    in Task 2.
    """

    def __init__(self, train, order):
        """
        self._counts: counts(w_i,w_(i-1))
        self._norm: counts(w_(i-1))
        self._tot_bigrams: total number of unique word bigram types
        self._follow_up: # of different words w_i following a given w_(i-1)
        self._preceding: # of different words w_(i-1) preceding a given w_i
        self._p_continuation: unigram continuation probability (see Task 2)
        self._lambda: normalising constant i.e. discounted probability mass
        """
        super().__init__(set(train), order)
        self._counts = lm.collections.defaultdict(float)
        self._norm = lm.collections.defaultdict(float)
        self._tot_bigrams = lm.collections.defaultdict(float)
        self._follow_up = lm.collections.defaultdict(float)
        self._preceding = lm.collections.defaultdict(float)
        self._p_continuation = lm.collections.defaultdict(float)
        self._lambda = lm.collections.defaultdict(float)
        # discount (0 < d < 1) obtained via offline parameter optimisation
        self._discount = 0.79
        """
        We loop over every pair of consecutive words in the dataset,
        updating self._counts, self._norm, self._follow_up,
        self._preceding and self._tot_bigrams
        """
        for i in range(self.order, len(train)):
            history = tuple(train[i - self.order + 1: i])
            word = train[i]
            self._counts[(word,) + history] += 1.0
            self._norm[history] += 1.0
            if self._counts[(word,) + history] == 1.0:
                self._tot_bigrams['grams'] += 1
                self._follow_up[history] += 1
                self._preceding[word] += 1
        """
        Once the total number of word bigram types is known, the continuation
        probability for each unigram is calculated. So is lambda.
        """
        for i in range(self.order, len(train)):
            history = tuple(train[i - self.order + 1: i])
            word = train[i]
            self._p_continuation[word] = self._preceding[word] / self._tot_bigrams['grams']
            self._lambda[history] = (self._discount / self._norm[history]) \
                                    * self._follow_up[history]

    """
    Method declarations
    """

    def counts(self, word_and_history):
        return self._counts[word_and_history]

    def norm(self, history):
        return self._norm[history]

    def bigrams(self, bigram):
        return self._tot_bigrams[bigram]

    def followup(self, history):
        return self._follow_up[history]

    def preceding(self, word):
        return self._uniques[word]

    def pcont(self, word):
        return self._p_continuation[word]

    def lambd(self, history):
        return self._lambda[history]

    """
    P_KN(w_i|w_(i-1)) calculated here
    """

    def probability(self, word, *history):
        if word not in self.vocab:
            return 0.0  # ensures language model only operates on defined vocab
        sub_history = tuple(history[-(self.order - 1):]) if self.order > 1 else ()
        norm = self.norm(sub_history)
        """
        Unseen bigrams p(x_i|x_(i-1)) are assigned back-off probability of last
        unigram p(x_i). In Kneser-Ney that is P_CONTINUATION(x_i).
        """
        if norm == 0:
            return self._p_continuation[word]
        else:  # P_KN(w_i|w_(i-1)) calculation
            return ((max(self._counts[(word,) + tuple(history)] - self._discount, 0)) \
                    / (self._norm[history])) + (self._p_continuation[word] * \
                                                self._lambda[history])


oov_train = lm.inject_OOVs(_snlp_train_song_words)  # injects probabilities for OOVs


def create_lm(vocab):
    KNB = KneserNeyBigram(oov_train, 2)  # bigram model
    """
    'Missing words' (in test set but not in training set) mapped to OOV token.
    """
    OOVAware = lm.OOVAwareLM(KNB, vocab - set(oov_train))

    return OOVAware

#! SETUP 3 (RIEDEL)
_snlp_test_dir = _snlp_book_dir + "/data/ohhla/dev"

#! SETUP 4 (RIEDEL)
_snlp_test_song_words = ohhla.words(ohhla.load_all_songs(_snlp_test_dir))
_snlp_test_vocab = set(_snlp_test_song_words)
_snlp_dev_vocab = set(_snlp_dev_song_words)
_snlp_train_vocab = set(_snlp_train_song_words)
_snlp_vocab = _snlp_test_vocab | _snlp_train_vocab | _snlp_dev_vocab
_snlp_lm = create_lm(_snlp_vocab)

#! ASSESSMENT 1 (RIEDEL)
_snlp_test_token_indices = [100, 1000, 10000]
_eps = 0.000001
for i in _snlp_test_token_indices:
    result = sum([_snlp_lm.probability(word, *_snlp_test_song_words[i-_snlp_lm.order+1:i]) for word in _snlp_vocab])
    print("Sum: {sum}, ~1: {approx_1}, <=1: {leq_1}".format(sum=result,
                                                            approx_1=abs(result - 1.0) < _eps,
                                                            leq_1=result - _eps <= 1.0))
#! ASSESSMENT 2 (RIEDEL)
lm.perplexity(_snlp_lm, _snlp_test_song_words)