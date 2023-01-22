from six.moves import xrange
import re
import unicodedata
import seaborn as sns
import spacy
from spacy.language import Language
from . import configuration_sys as conf_sys


def get_colors(n=5, mode='hex'):
    palette = sns.color_palette(None, n)
    if mode == 'hex':
        return map(lambda rgb: '#%02x%02x%02x' % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)),
                   palette)
    else:
        raise ValueError(mode)


def slugify(value, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace.
    """
    if value in conf_sys.ANY_ENTITY_MARKERS:
        return conf_sys.ANY_ENTITY_TOKEN
    new_value = str(value)
    if allow_unicode:
        new_value = unicodedata.normalize('NFKC', new_value)
    else:
        new_value = unicodedata.normalize('NFKD', new_value).encode('ascii', 'ignore').decode('ascii')
    new_value = re.sub(r'[^\w\s-]', '', new_value).strip()
    new_value = re.sub(r'[-\s_]+', '_', new_value)
    if new_value == '':
        new_value = value
    return new_value


def length_preserving_clean_text(string):
    """
    Perform cleaning operations that DO NOT change the original text size
    """
    string = string.replace(u'\xa0', u' ').replace("’", "'")
    return string


def make_ngrams(s, n):
    # s = u'{t}{s}{t}'.format(s=safe_unicode(s), t=('$' * (n - 1)))
    n = len(s) if len(s) < n else n
    return (s[i:i + n] for i in xrange(len(s) - n + 1))


def path_2_text(text_path):
    """
    Generates the content of the text path
    Returns the original text and the cleaned text (only length-preserving modifications)
    """
    # Read and apply basic length-preserving text processing
    original_text = open(text_path, 'r', encoding='utf-8').read()
    cleaned_text = length_preserving_clean_text(original_text)
    return original_text, cleaned_text


def paths_to_text(text_paths):
    """
    Generates the content of each text path
    """
    for text_path in text_paths:
        yield open(text_path, 'r', encoding='utf-8').read()


LIST_ITEMS = {'-', '*'}


def conditional_component_decorator(function):
    """
    Adds a "set_custom_boundaries" decorator on the function (set_custom_boundaries)
    if spacy is in version >= 3.0
    For version under 3, the decorator does not exist and should not be added
    """
    if int(spacy.__version__[0]) >= 3:
        return Language.component('set_custom_boundaries', func=function)
    else:
        return function


@conditional_component_decorator
def set_custom_boundaries(doc):
    previous_is_newline = False
    for i, token in enumerate(doc):
        if token.text.startswith('\n  '):
            doc[i].is_sent_start = True
            previous_is_newline = True
        elif '\n' in token.text:
            previous_is_newline = True
        else:
            # List item -> new sentence
            if token.text in LIST_ITEMS and previous_is_newline:
                doc[i - 1].is_sent_start = True
            # Capitalized and just after newline -> new sentence
            if token.is_title and previous_is_newline:
                doc[i - 1].is_sent_start = True
                # Upper case and just after newline and previous token is not uppercased -> new sentence
            if token.is_upper and previous_is_newline and i - 2 >= 0 and not doc[i - 2].is_upper:
                doc[i - 1].is_sent_start = True
            # Just after newline and

            previous_is_newline = False

    return doc


class Intervals(object):
    def __init__(self, is_a_hierarchy):
        self.intervals = []
        self.is_a_hierarchy = is_a_hierarchy

    def _is_overlapping_intervals(self, a, b):
        if not any([va == vb or va in self.is_a_hierarchy.get(vb, []) or vb in self.is_a_hierarchy.get(va, [])
                    for va in a[2] for vb in b[2]]):
            return False
        if b[0] < a[1] and b[1] > a[0]:
            return True
        elif a[0] < b[1] and a[1] > b[0]:
            return True
        else:
            return False

    def __contains__(self, interval):
        return any(
            self._is_overlapping_intervals(interval, other)
            for other in self.intervals
        )

    def append(self, interval):
        self.intervals.append(interval)
