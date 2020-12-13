import nltk, ssl
from nltk.corpus import brown
from collections import Counter
import numpy as np
from random import choice

DUMMY_TRAINING_SET = [[('The', 'AT'), ('dog', 'NN-TL'), ('jumped', 'VBD'), ('over', 'IN'), ('the', 'AT'), ('fence', 'NN'), ('.', '.')]]
DUMMY_TEST_SET =  [[('The', 'AT'), ('the', 'AT'), ('the', 'NN'), ('unknown', 'NN')]]

def load_training_test_sets():
    sents = list(brown.tagged_sents(categories='news'))
    slice_idx = int(0.9*len(sents))
    training_set = sents[:slice_idx]
    test_set = sents[slice_idx+1:]
    return training_set, test_set

def load_dummy_sets():
    return DUMMY_TRAINING_SET, DUMMY_TEST_SET

class HmmTagger:

    tagged_sentences = None
    tagged_words = None
    tags_counter = Counter()
    tags_pairs_counter = {}


    def __init__(self, training_set):
        self.tagged_sentences = training_set
        self.tagged_words = [tagged_word for tagged_sentence in training_set for tagged_word in tagged_sentence]
        self.tags_counter = Counter(i[1] for i in self.tagged_words)
        self.initial_tags_counter = Counter(tagged_sentence[0][1] for tagged_sentence in training_set)
        self.__count_tags_pairs()


    def __count_tags_pairs(self):
        self.__init_tags_pairs_counter()
        for i in range(1, len(self.tagged_words)):
            preceding_word_tag = self.tagged_words[i-1][1]
            current_word_tag = self.tagged_words[i][1]
            self.tags_pairs_counter[preceding_word_tag][current_word_tag] = \
                self.tags_pairs_counter[preceding_word_tag][current_word_tag] + 1


    def __init_tags_pairs_counter(self):
        for row_tag in self.tags_counter.keys():
            self.tags_pairs_counter[row_tag] = {}
            for col_tag in self.tags_counter.keys():
                self.tags_pairs_counter[row_tag][col_tag] = 0


    def get_initial_tag_probability(self, tag):
        return self.initial_tags_counter[tag]/len(self.tagged_sentences)

    """
    returns P(t_i|t_i-1) - probablity of current tag to appear given previous tag
    """
    def get_transition_probability(self, preceding_tag, current_tag):
        return self.tags_pairs_counter[preceding_tag][current_tag]/self.tags_counter[preceding_tag]

    """
    returns P(word|tag) - probability of word to appear given current tag.
    """
    def get_emmission_probability(self, tag, word):
        word_tags_counter = Counter([tagged_word[1] for tagged_word in self.tagged_words if tagged_word[0] == word])
        if tag not in word_tags_counter.keys():
            return 0
        return word_tags_counter[tag]/self.tags_counter[tag]


    def is_word_known(self, word):
        return word in (tagged_word[0] for tagged_word in self.tagged_words)


def get_transition_probabilities(hmmTagger, tags, current_tag):
    probs = []
    for prev_tag in tags:
        probs.append(hmmTagger.get_transition_probability(prev_tag, current_tag))
    return np.array(probs)


def viterbi(words, tags, hmmTagger):
    tags = list(sorted(tags))
    trellis = np.zeros((len(tags), len(words)), dtype=np.float)
    back_pointers = np.zeros((len(tags), len(words)), dtype=np.int)
    best_path = []
    for i, tag in enumerate(tags):
        trellis[i,0] = hmmTagger.get_initial_tag_probability(tag) * hmmTagger.get_emmission_probability(tag, words[0])
    for word_ind in range(1, len(words)):
        for tag_ind, tag in enumerate(tags):
            if (hmm_tagger.is_word_known(words[word_ind])):
                max_prev_word_tag_ind = np.argmax(trellis[:, word_ind - 1] * get_transition_probabilities(hmmTagger, tags,tag) * hmmTagger.get_emmission_probability(tag, words[word_ind]))
            else:
                max_prev_word_tag_ind = choice(range(len(tags)))
            max_prev_tag = tags[max_prev_word_tag_ind]
            back_pointers[tag_ind, word_ind] = max_prev_word_tag_ind
            trellis[tag_ind, word_ind] = trellis[max_prev_word_tag_ind, word_ind-1] * hmmTagger.get_transition_probability(max_prev_tag, tag) * hmmTagger.get_emmission_probability(tag, words[word_ind])
    best_path.insert(0,tags[np.argmax(trellis[:,len(words)-1])])
    for word_ind in range(len(words)-1,0, -1):
        max_tag_ind = np.argmax(trellis[:,word_ind])
        best_path.insert(0,tags[back_pointers[max_tag_ind,word_ind]])
    return best_path


def test_hmm_tagger():
    training_set, test_set = load_dummy_sets()
    hmm_tagger = HmmTagger(training_set)
    tagged_words = [tagged_word for tagged_sentence in training_set for tagged_word in tagged_sentence]
    tags = set(i[1] for i in tagged_words)
    # for preceding_tag in tags:
    #     for current_tag in tags:
    #         transition_probablity = hmm_tagger.get_transition_probability(preceding_tag, current_tag)
    #         print("({},{}): {}".format(preceding_tag, current_tag, transition_probablity))
    # the_emission = hmm_tagger.get_emmission_probability('AT', 'the')
    # print('the emission: {}'.format(the_emission))
    print(hmm_tagger.get_initial_tag_probability('AT'))
    print(hmm_tagger.get_emmission_probability('AT', 'The'))
    test_sequence = ['The', 'dog','jumped','over','the','fence','.']
    prediction = viterbi(test_sequence, tags, hmm_tagger)
    print(prediction)


if __name__ == "__main__":
    # test_hmm_tagger()
    training_set, test_set = load_training_test_sets()
    tagged_words = [tagged_word for tagged_sentence in training_set for tagged_word in tagged_sentence]
    tags = set(i[1] for i in tagged_words)
    hmm_tagger = HmmTagger(training_set)
    test_sentence = [('But', 'CC'), ('by', 'IN'), ('the', 'AT'), ('end', 'NN'), ('of', 'IN'), ('the', 'AT'), ('three-month', 'JJ'), ('period', 'NN'), (',', ','), ('in', 'IN'), ('October', 'NP'), ('1960', 'CD'), (',', ','), ('something', 'PN'), ('approaching', 'VBG'), ('calm', 'JJ'), ('settled', 'VBN'), ('on', 'IN'), ('the', 'AT'), ('Congo', 'NP'), ('.', '.')]
    prediction = viterbi([tagged_word[0] for tagged_word in test_sentence], tags, hmm_tagger)
    print(test_sentence)
    print(prediction)
