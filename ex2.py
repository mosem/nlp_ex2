import nltk, ssl
from nltk.corpus import brown
from collections import Counter

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


def most_likely_tag_baseline():
    training_set, test_set = load_training_test_sets()
    # training_set, test_set = load_dummy_sets()
    mle_dict = compute_mle(training_set)
    compute_error_rate(test_set, mle_dict)


if __name__ == "__main__":
    print('running most likely tag baseline')
    most_likely_tag_baseline()