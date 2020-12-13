import nltk, ssl
from nltk.corpus import brown
from collections import Counter
from hmm_tagger import HmmTagger
import numpy as np
from random import choice

DUMMY_TRAINING_SET = [[('The', 'AT'), ('dog', 'NN-TL'), ('jumped', 'VBD'), ('over', 'IN'), ('the', 'AT'), ('fence', 'NN'), ('.', '.')]]
DUMMY_TEST_SET =  [[('The', 'AT'), ('the', 'AT'), ('the', 'NN'), ('unknown', 'NN')]]

def download_corpus():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('brown')

def load_dummy_sets():
    return DUMMY_TRAINING_SET, DUMMY_TEST_SET

def load_training_test_sets():
    sents = list(brown.tagged_sents(categories='news'))
    slice_idx = int(0.9*len(sents))
    training_set = sents[:slice_idx]
    test_set = sents[slice_idx+1:]
    return training_set, test_set

def compute_mle(training_set):
    mle_dict = dict()
    tagged_words = [tagged_word for tagged_sentence in training_set for tagged_word in tagged_sentence]
    print(tagged_words)
    f = open('tagged_words.txt', 'w')
    f.write('\n'.join('{} {}'.format(x[0],x[1]) for x in tagged_words))
    f.close()
    words_set = set(i[0]for i in tagged_words)
    for word in words_set:
        tags = [tagged_word[1] for tagged_word in tagged_words if tagged_word[0] == word]
        counter = Counter(tags)
        # print('{}, {}'.format(word, tags))
        mle_dict[word] = counter.most_common(1)[0][0]
    return mle_dict

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
            if (hmmTagger.is_word_known(words[word_ind])):
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

def compute_error_rate(test_set,training_mle):
    tagged_words = [tagged_word for tagged_sentence in test_set for tagged_word in tagged_sentence]
    words,tags = zip(*tagged_words)
    words_set = set(i[0] for i in tagged_words)
    total_incorrect_known_words_counter = 0
    total_known_words_counter = 0
    total_incorrect_unknown_words_counter = 0
    total_unknown_words_counter = 0
    for word in words_set:
        current_tag = [tagged_word[1] for tagged_word in tagged_words if tagged_word[0] == word]
        if word in training_mle.keys():
            incorrect_counter = len(current_tag) - current_tag.count(training_mle[word])
            total_incorrect_known_words_counter = total_incorrect_known_words_counter + incorrect_counter
            total_known_words_counter = total_known_words_counter + words.count(word)
        else:
            incorrect_counter = len(current_tag) - current_tag.count('NN')
            total_incorrect_unknown_words_counter = total_incorrect_unknown_words_counter + incorrect_counter
            total_unknown_words_counter = total_unknown_words_counter + words.count(word)

    known_words_error_rate = total_incorrect_known_words_counter/total_known_words_counter
    if total_unknown_words_counter == 0:
        unknown_words_error_rate = 0
    else:
        unknown_words_error_rate = total_incorrect_unknown_words_counter/total_unknown_words_counter
    total_error_rate = (total_incorrect_known_words_counter+total_incorrect_unknown_words_counter)/len(words)

    print('known words error rate: {}'.format(known_words_error_rate))
    print('unknown words error rate: {}'.format(unknown_words_error_rate))
    print('total error rate: {}'.format(total_error_rate))
    return


def most_likely_tag_baseline(training_set, test_set):
    mle_dict = compute_mle(training_set)
    compute_error_rate(test_set, mle_dict)


def evaluate_vanilla_viterbi(training_set, test_set):
    hmm_tagger = HmmTagger(training_set)
    tagged_words = [tagged_word for tagged_sentence in training_set for tagged_word in tagged_sentence]
    words, tags = zip(*tagged_words)
    words_set = set(i[0] for i in tagged_words)
    tags = set(i[1] for i in tagged_words)

    total_incorrect_known_words_counter = 0
    total_known_words_counter = 0
    total_incorrect_unknown_words_counter = 0
    total_unknown_words_counter = 0

    for sentence in test_set:
        sentence_words, sentence_tags = zip(*sentence)
        prediction = viterbi(sentence_words, tags, hmm_tagger)
        for i in range(len(sentence)):
            if sentence_words[i] in words_set:
                total_known_words_counter = total_known_words_counter + 1
                if prediction[i] != sentence_tags[i]:
                    total_incorrect_known_words_counter = total_incorrect_known_words_counter + 1
            else:
                total_unknown_words_counter = total_unknown_words_counter + 1
                if prediction[i] != sentence_tags[i]:
                    total_incorrect_unknown_words_counter = total_incorrect_unknown_words_counter + 1

    known_words_error_rate = total_incorrect_known_words_counter / total_known_words_counter
    if total_unknown_words_counter == 0:
        unknown_words_error_rate = 0
    else:
        unknown_words_error_rate = total_incorrect_unknown_words_counter / total_unknown_words_counter
    total_error_rate = (total_incorrect_known_words_counter + total_incorrect_unknown_words_counter) / len(words)

    print('known words error rate: {}'.format(known_words_error_rate))
    print('unknown words error rate: {}'.format(unknown_words_error_rate))
    print('total error rate: {}'.format(total_error_rate))


if __name__ == "__main__":
    # training_set, test_set = load_dummy_sets()
    training_set, test_set = load_training_test_sets()

    # print('running most likely tag baseline')
    # most_likely_tag_baseline(training_set, test_set)

    print('running vanilla viterbi')
    evaluate_vanilla_viterbi(training_set, test_set)