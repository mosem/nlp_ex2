import nltk, ssl
from nltk.corpus import brown
from collections import Counter
from hmm_tagger import HmmTagger
import pandas as pd

from dataset_utils import *

def compute_mle(training_set):
    mle_dict = dict()
    word_tag_counter = dict()
    tagged_words = [tagged_word for tagged_sentence in training_set for tagged_word in tagged_sentence]
    for word, tag in tagged_words:
        if word not in word_tag_counter.keys():
            word_tag_counter[word] = Counter()
        word_tag_counter[word][tag] += 1

    for word in word_tag_counter.keys():
        mle_dict[word] = word_tag_counter[word].most_common(1)[0][0]

    return mle_dict


def compute_error_rate_mle(test_set,training_mle):
    known_words = set(training_mle.keys())

    total_incorrect_known_words_counter = 0
    total_known_words_counter = 0
    total_incorrect_unknown_words_counter = 0
    total_unknown_words_counter = 0
    total_words_counter = 0

    for sentence in test_set:
        for word,tag in sentence:
            total_words_counter += 1
            if word in known_words:
                total_known_words_counter += 1
                if tag != training_mle[word]:
                    total_incorrect_known_words_counter += 1
            else:
                total_unknown_words_counter += 1
                if tag != 'NN':
                    total_incorrect_unknown_words_counter += 1

    known_words_error_rate = total_incorrect_known_words_counter/total_known_words_counter
    if total_unknown_words_counter == 0:
        unknown_words_error_rate = 0
    else:
        unknown_words_error_rate = total_incorrect_unknown_words_counter/total_unknown_words_counter
    total_error_rate = (total_incorrect_known_words_counter+total_incorrect_unknown_words_counter)/total_words_counter

    print('known words error rate: {}'.format(known_words_error_rate))
    print('unknown words error rate: {}'.format(unknown_words_error_rate))
    print('total error rate: {}'.format(total_error_rate))


def most_likely_tag_baseline(training_set, test_set):
    mle_dict = compute_mle(training_set)
    compute_error_rate_mle(test_set, mle_dict)


def compute_error_rate(training_set, test_set, hmm_tagger):
    tagged_words = [tagged_word for tagged_sentence in training_set for tagged_word in tagged_sentence]
    known_words = set(i[0] for i in tagged_words)

    total_incorrect_known_words_counter = 0
    total_known_words_counter = 0
    total_incorrect_unknown_words_counter = 0
    total_unknown_words_counter = 0

    total_test_words = 0
    for sentence in test_set:
        sentence_words, sentence_tags = zip(*sentence)
        total_test_words += len(sentence)
        prediction = hmm_tagger.viterbi(sentence_words)
        for i in range(len(sentence)):
            if sentence_words[i] in known_words:
                total_known_words_counter += 1
                if prediction[i] != sentence_tags[i]:
                    total_incorrect_known_words_counter += 1
            else:
                total_unknown_words_counter += 1
                if prediction[i] != sentence_tags[i]:
                    total_incorrect_unknown_words_counter += 1

    known_words_error_rate = total_incorrect_known_words_counter / total_known_words_counter
    if total_unknown_words_counter == 0:
        unknown_words_error_rate = 0
    else:
        unknown_words_error_rate = total_incorrect_unknown_words_counter / total_unknown_words_counter
    total_error_rate = (total_incorrect_known_words_counter + total_incorrect_unknown_words_counter) / total_test_words

    print('known words error rate: {}'.format(known_words_error_rate))
    print('unknown words error rate: {}'.format(unknown_words_error_rate))
    print('total error rate: {}'.format(total_error_rate))


def evaluate_vanilla_viterbi(training_set, test_set):
    print('initiating vanilla hmm tagger.')
    hmm_tagger = HmmTagger(training_set)
    print('done training hmm tagger.')
    compute_error_rate(training_set, test_set, hmm_tagger)


def evaluate_smoothing_viterbi(training_set, test_set):
    print('initiating one smoothing hmm tagger.')
    hmm_tagger = HmmTagger(training_set,smoothing=True)
    print('done training hmm tagger.')
    compute_error_rate(training_set, test_set, hmm_tagger)


def evaluate_psuedowords_viterbi(training_set, test_set):
    print('initiating psuedowords hmm tagger.')
    hmm_tagger = HmmTagger(training_set, use_psuedowords=True)
    print('done training hmm tagger.')
    compute_error_rate(training_set, test_set, hmm_tagger)


def evaluate_psuedowords_smoothing_viterbi(training_set, test_set):
    print('initiating psuedowords hmm tagger.')
    hmm_tagger = HmmTagger(training_set, smoothing=True, use_psuedowords=True)
    print('done training hmm tagger.')
    compute_error_rate(training_set, test_set, hmm_tagger)

    print('computing confusion matrix')
    get_confusion_matrix(training_set, test_set, hmm_tagger).to_csv("confusion_matrix.csv")


def get_confusion_matrix(training_set, test_set, hmm_tagger):
    tags = sorted(set([tagged_word[1] for tagged_sentence in training_set for tagged_word in tagged_sentence]) | \
                    set([tagged_word[1] for tagged_sentence in test_set for tagged_word in tagged_sentence]))

    confusion_matrix = {tag1: {tag2: 0 for tag2 in tags} for tag1 in tags}

    for sentence in test_set:
        sentence_words, sentence_tags = zip(*sentence)
        prediction = hmm_tagger.viterbi(sentence_words)
        for i, true_tag in enumerate(sentence_tags):
            predicted_tag = prediction[i]
            confusion_matrix[true_tag][predicted_tag] += 1

    return pd.DataFrame(confusion_matrix)



if __name__ == "__main__":
    # training_set, test_set = load_dummy_sets()
    training_set, test_set = load_training_test_sets() # returns datasets after tags are cleaned.

    print('running most likely tag baseline')
    most_likely_tag_baseline(training_set, test_set)
    print('================================\n\n')

    print('running vanilla viterbi')
    evaluate_vanilla_viterbi(training_set, test_set)
    print('================================\n\n')

    print('running smoothing viterbi')
    evaluate_smoothing_viterbi(training_set, test_set)
    print('================================\n\n')

    print('running psuedowords viterbi')
    evaluate_psuedowords_viterbi(training_set, test_set)
    print('================================\n\n')

    print('running psuedowords smoothing viterbi')
    evaluate_psuedowords_smoothing_viterbi(training_set, test_set)
    print('================================\n\n')

    print('done.')