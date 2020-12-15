import nltk, ssl
from nltk.corpus import brown
from collections import Counter
from psuedowords_utils import *

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
    sents = clean_dataset(list(brown.tagged_sents(categories='news')))
    slice_idx = int(0.9*len(sents))
    training_set = sents[:slice_idx]
    test_set = sents[slice_idx+1:]
    return training_set, test_set

def clean_dataset(dataset):
    new_dataset = dataset
    for i, tagged_sentence in enumerate(dataset):
        for j, (word, tag) in enumerate(tagged_sentence):
            new_dataset[i][j] = (word, clean_tag(tag))
    return new_dataset


def clean_tag(tag):
    return tag.split('[\+-]')[0]

def __convert_training_set(training_set, low_freq_threshold):
    new_training_set = training_set
    words_counter = Counter([word_tag[0] for sentence in new_training_set for word_tag in sentence])
    for i, tagged_sentence in enumerate(new_training_set):
        for j, (word,tag) in enumerate(tagged_sentence):
            if words_counter[word] < low_freq_threshold:
                new_training_set[i][j] = (get_psuedoword(word, j==0), tag)
    return new_training_set