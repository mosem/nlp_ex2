import unittest

import ex2


class q4a(unittest.TestCase):
    def test_downloading_corpus(self):
        ex2.download_corpus()
        train_set, test_set = ex2.load_training_test_sets()
        train_count, test_count = len(train_set), len(test_set)
        all_count = train_count + test_count
        train_ratio, test_ratio = train_count / all_count, test_count / all_count
        train_ratio, test_ratio = round(train_ratio,1), round(test_ratio,1)
        self.assertEqual(train_ratio, 0.9)
        self.assertEqual(test_ratio, 0.1)

class q4b(unittest.TestCase):
    def test_i(self):
        pass
    def test_ii(self):
        pass

class q4c(unittest.TestCase):
    def test_viterbi(self):
        pass



def test_a():
    print(f'*****testing a')
    ex2.download_corpus()
    train_set, test_set = ex2.load_training_test_sets()
    train_count, test_count = len(train_set), len(test_set)
    all_count = train_count + test_count
    print(f'Training set size is: {train_count}, test set size is: {test_count}.'
          f'\nThe training set is {train_count / all_count * 100}% and test set is {test_count / all_count * 100}')


def run_all_tests():
    test_a()


if __name__ == '__main__':
    print(f'******running all tests')
    # run_all_tests()
    unittest.main()