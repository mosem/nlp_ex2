from nltk.corpus import brown
from collections import Counter
import numpy as np
from random import choice
from psuedowords_utils import *

DUMMY_TRAINING_SET = [[('The', 'AT'), ('dog', 'NN-TL'), ('jumped', 'VBD'), ('over', 'IN'), ('the', 'AT'), ('fence', 'NN'), ('.', '.')]]
DUMMY_TEST_SET =  [[('The', 'AT'), ('the', 'AT'), ('the', 'NN'), ('unknown', 'NN'), ('555', 'IN')]]

def load_training_test_sets():
    sents = list(brown.tagged_sents(categories='news'))
    slice_idx = int(0.9*len(sents))
    training_set = sents[:slice_idx]
    test_set = sents[slice_idx+1:]
    return training_set, test_set

def load_dummy_sets():
    return DUMMY_TRAINING_SET, DUMMY_TEST_SET


class HmmTagger:

    def __init__(self, training_set,smoothing=False, use_psuedowords=False, low_freq_threshold=5):

        self.smoothing = smoothing
        self.use_psuedowords = use_psuedowords
        self.low_freq_threshold = low_freq_threshold

        self.training_set = self.__convert_training_set(training_set) if use_psuedowords else training_set

        words, tags = zip(*[tagged_word for sentence in self.training_set for tagged_word in sentence])

        self.known_words = set(words)
        self.known_tags = set(tags)

        self.__init_tables()


    def __convert_training_set(self, training_set):
        new_training_set = training_set
        self.words_counter = Counter([word_tag[0] for sentence in new_training_set for word_tag in sentence])
        for i, tagged_sentence in enumerate(new_training_set):
            for j, (word,tag) in enumerate(tagged_sentence):
                if self.words_counter[word] < self.low_freq_threshold:
                    new_training_set[i][j] = (get_psuedoword(word, j==0), tag)
        return new_training_set


    def __init_tables(self):
        self.transition_table = self.__init_transition_table()
        self.emission_table = self.__init_emission_table()
        for sentence in self.training_set:
            temp_sentence = [(None,'*')] + sentence + [(None,'STOP')]
            for i in range(1, len(temp_sentence)):
                preceding_tag = temp_sentence[i-1][1]
                current_tag = temp_sentence[i][1]
                self.transition_table[preceding_tag][current_tag] += 1
            for i, (word, tag) in enumerate(sentence):
                self.emission_table[tag][word] += 1

        if self.smoothing:
            self.__update_add_one_smoothing_on_emission_table()

        self.__normalize_table(self.transition_table)
        self.__normalize_table(self.emission_table)


    def __init_transition_table(self):
        table = {}
        tags = self.known_tags | {'*', 'STOP'}
        for row_tag in tags:
            table[row_tag] = {}
            for col_tag in tags:
                table[row_tag][col_tag] = 0
        return table


    def __init_emission_table(self):
        table = {}
        words = self.known_words | set(map(str, Category)) if self.use_psuedowords else self.known_words
        for tag in self.known_tags:
            table[tag] = {}
            for word in words:
                table[tag][word] = 0
        return table


    def __update_add_one_smoothing_on_emission_table(self):
        words = self.known_words | set(map(str, Category)) if self.use_psuedowords else self.known_words
        for tag in self.known_tags:
            for word in words:
                self.emission_table[tag][word] += 1


    """
        normalize dictionary of dictionaries
        each dictinoary represents a row in a table
        this function normalizes table by row.
    """
    def __normalize_table(self, table):
        for row_key, row_dict in table.items():
            total_row_sum = sum(row_dict.values())
            for col_key, value in row_dict.items():
                if total_row_sum != 0:
                    table[row_key][col_key] = value/total_row_sum


    """
    returns P(t_i|t_i-1) - probablity of current tag to appear given previous tag
    """
    def get_transition_probability(self, preceding_tag, current_tag):
        return self.transition_table[preceding_tag][current_tag]


    """
    returns P(word|tag) - probability of word to appear given current tag.
    """
    def get_emission_probability(self, tag, word, is_first_word=False):
        if self.use_psuedowords:
            if not self.is_word_known(word) or self.words_counter[word] <= self.low_freq_threshold:
                word = get_psuedoword(word, is_first_word)
        elif not self.is_word_known(word):
            return 0
        return self.emission_table[tag][word]


    def is_word_known(self, word):
        return word in self.known_words


    def get_transition_probabilities(self, tags_list, current_tag):
        probs = []
        for prev_tag in tags_list:
            probs.append(self.get_transition_probability(prev_tag, current_tag))
        return np.array(probs)


    def viterbi(self, sentence):
        tags_list = list(sorted(self.known_tags))
        viterbi_table = np.zeros((len(tags_list), len(sentence)), dtype=np.float)
        back_pointers = np.zeros((len(tags_list), len(sentence)), dtype=np.int)
        best_path = []
        for i, tag in enumerate(tags_list):
            viterbi_table[i,0] = self.get_transition_probability('*', tag) * self.get_emission_probability(tag, sentence[0], is_first_word=True)
        for word_ind in range(1, len(sentence)):
            word = sentence[word_ind]
            for tag_ind, tag in enumerate(tags_list):
                if self.is_word_known(word) or self.use_psuedowords:
                    max_prev_word_tag_ind = np.argmax(viterbi_table[:, word_ind - 1] * self.get_transition_probabilities(tags_list, tag) * self.get_emission_probability(tag, word))
                else:
                    max_prev_word_tag_ind = choice(range(len(tags_list)))
                max_prev_tag = tags_list[max_prev_word_tag_ind]
                back_pointers[tag_ind, word_ind] = max_prev_word_tag_ind
                viterbi_table[tag_ind, word_ind] = viterbi_table[max_prev_word_tag_ind, word_ind - 1] * self.get_transition_probability(max_prev_tag, tag) * self.get_emission_probability(tag, sentence[word_ind])
        transitions_to_stop = [viterbi_table[tag_ind,-1]*self.get_transition_probability(tags_list[tag_ind], 'STOP') for tag_ind in range(len(tags_list))]
        best_path.insert(0, tags_list[np.argmax(transitions_to_stop)])
        for word_ind in range(len(sentence) - 1, 0, -1):
            max_tag_ind = np.argmax(viterbi_table[:, word_ind])
            best_path.insert(0, tags_list[back_pointers[max_tag_ind, word_ind]])
        return best_path


def test_hmm_tagger():
    # training_set, test_set = load_training_test_sets()
    training_set, test_set = load_dummy_sets()
    hmm_tagger = HmmTagger(training_set, use_psuedowords=True)
    training_words, training_tags = zip(*choice(training_set))
    test_words, test_tags = zip(*choice(test_set))
    for k,v in hmm_tagger.emission_table.items():
        print(k,v)
    # test_sequence = ['The', 'dog','jumped','over','the','fence','.']
    print(training_words)
    print(training_tags)

    prediction = hmm_tagger.viterbi(training_words)

    print(prediction)



if __name__ == "__main__":
    test_hmm_tagger()

