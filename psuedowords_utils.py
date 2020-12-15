import re
from enum import Enum

class Category(Enum):

    def __str__(self):
        return str(self.value)

    OTHER = '_OTHER'
    TWO_DIGIT_NUM = '_TWO_DIGIT_NUM'
    FOUR_DIGIT_NUM = '_FOUR_DIGIT_NUM'
    CONTAINS_DIGIT_AND_DASH_OR_SLASH = '_CONTAINS_DIGIT_AND_DASH_OR_SLASH'
    CONTAINS_DIGIT_AND_COMMA = '_CONTAINS_DIGIT_AND_COMMA'
    CONTAINS_DIGIT_AND_PERIOD = '_CONTAINS_DIGIT_AND_PERIOD'
    OTHER_NUM = '_OTHER_NUM'
    ALL_CAPS = '_ALL_CAPS'
    CAP_PERIOD = '_CAP_PERIOD'
    FIRST_WORD = '_FIRST_WORD'
    INIT_CAP = '_INIT_CAP'
    LOWER_CASE = '_LOWER_CASE'
    NON_LETTER = '_NON_LETTER'


def get_psuedoword(word, is_first_word=False):
    if any(char.isdigit() for char in word):
        if re.compile(r'^\d\d$').search(word):
            return Category.TWO_DIGIT_NUM.value
        if re.compile(r'^\d\d\d\d$').search(word):
            return Category.FOUR_DIGIT_NUM.value
        if word.find('-') or word.find('/'):
            return Category.CONTAINS_DIGIT_AND_DASH_OR_SLASH.value
        if word.find('.'):
            return Category.CONTAINS_DIGIT_AND_PERIOD.value
        if word.find(','):
            return Category.CONTAINS_DIGIT_AND_COMMA.value
        if re.compile(r'^\d*$').search(word):
            return Category.OTHER_NUM.value
    if is_first_word:
        return Category.FIRST_WORD.value
    if re.compile(r'^[A-Z]*$').search(word):
        return Category.ALL_CAPS.value
    if re.compile(r'^[A-Z]\.$').search(word):
        return Category.CAP_PERIOD.value
    if word[0].isupper():
        return Category.INIT_CAP.value
    if word[0].islower():
        return Category.LOWER_CASE.value
    if re.compile(r'^\W$').search(word):
        return Category.NON_LETTER.value

    return Category.OTHER.value