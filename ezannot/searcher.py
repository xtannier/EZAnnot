import datetime
import logging
import re
import sys
import math
from copy import deepcopy

import pandas as pd
import spacy
from spacy import symbols, tokens
from nltk.stem.snowball import FrenchStemmer
from simstring.database.dict import DictDatabase
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.searcher import Searcher
from six.moves import xrange
from tqdm import tqdm

from . import constants
from . import schema
# project modules
from . import toolbox
from .brat import to_brat_entity
from .configuration_sys import EXPRESSION_TYPES, APPROX_MATCHING_LENGTH_MIN, TERM, ENTITY, REGEX, EXCLUSION, \
    TERMINO_COLUMN_NAMES, TERMINO_HEADER_HEIGHT, DISABLED_TERMS, TERMINO_ID, EXPRESSION, EXPR_TYPE, PREANNOTATION, \
    ANY_ENTITY_TOKEN
from .preprocess import SectionSplitter
from .rules import check_rule, protect_rule
from .toolbox import slugify, path_2_text, set_custom_boundaries, length_preserving_clean_text

# Pathos is the fork of multiprocessing that uses dill.
# dill can serialize almost anything in python, including object method, which is necessary here
# Also, we use _ProcessPool instead of ProcessPool in order to be able to use keyword arguments
# (see https://stackoverflow.com/questions/39746758/how-to-pass-keywords-list-to-pathos-multiprocessing)
# from pathos.multiprocessing import _ProcessPool as Pool


SPACE_TOKENIZER_REGEX = re.compile('(\S+)')
NUM_REGEX = re.compile('\d+([,.]\d+)?')
JOKER_REGEX = re.compile('<JOKER_(.+)_(.+)>')
# wildcard inserted into the trie
JOKER = '<JOKER>'
# Prefix for terms created internally (not by the human, not seen in final annotations)
INTERN_TERM_PREFIX = 'INTERN_TERM_'


def termino_error_message(file, index, message):
    return '{} - {} - {}'.format(file, index + TERMINO_HEADER_HEIGHT + 1, message)


class DictionaryLookup(object):
    def __init__(self):
        self.entities = set()

    def add(self, element):
        self.entities.add(element)

    def search(self, element):
        if element not in self.entities:
            return []
        else:
            return [element]


class Joker(object):
    def __init__(self, min_number, max_number):
        self.min_number = min_number
        self.max_number = max_number


class Term(object):
    def __init__(self, text, start, end, from_terminology=False, from_match=None, stopper=False):
        self.text = text
        self.start = start
        self.end = end
        self.from_terminology = from_terminology
        self.from_match = from_match
        if self.from_match is not None and not self.from_terminology:
            raise ValueError('If the term has been built from a match, it must be "from terminology"')
        self.stopper = stopper

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if self.from_terminology:
            if self.from_match:
                return 'Term(<{}_{}>,{},{})'.format(self.text, self.from_match, self.start, self.end)
            else:
                return 'Term(<{}>,{},{})'.format(self.text, self.start, self.end)
        else:
            return 'Term("{}",{},{})'.format(self.text, self.start, self.end)


class TerminologyTrie(object):
    END = '_end_'
    STRICT_MODE = 0
    MATCH_MODE = 1

    def __init__(self, conf):
        self.root = dict()
        self.conf = conf

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '{}'.format(self.root)

    def keys(self):
        return self.root.keys()

    def add(self, term_list, value):
        '''
        @param term_list list of strings
        '''
        current_dict = self.root
        for term in term_list:
            current_dict = current_dict.setdefault(term, {})
        end_values = current_dict.get(TerminologyTrie.END, [])
        end_values.append(value)
        current_dict[TerminologyTrie.END] = end_values

    # def search(self, term_list):
    #     '''
    #     @param term_list list of objects of type Term
    #     '''
    #     current_dict = self.root
    #     for term in term_list:
    #         if term.text in current_dict:
    #             return False
    #         current_dict = current_dict[term.text]
    #     return current_dict.get(TerminologyTrie.END, False)

    def match(self, term_list):
        '''
        @param term_list list of objects of type Term
        '''
        current_dict = self.root
        start = term_list[0].start
        for term in term_list:
            if term.text not in current_dict:
                break
            current_dict = current_dict[term.text]
            if TerminologyTrie.END in current_dict:
                for r in current_dict[TerminologyTrie.END]:
                    yield r, start, term.end

    def _match_from(self, term_list, start, previous_end, current_end, from_index, current_index, current_dict,
                    matches_met=None,
                    jokers=(0, -1), debug_level=0, iteration_level=0):
        '''
        @param term_list list of objects of type Term
        @param matches_met list of the matches from a previous iteration that have been met so far.
               Useful to keep trace of links between matches (a match has been created with another previous match inside)
        '''
        if current_index < len(term_list):
            terms = term_list[current_index]
            min_joker_number, max_joker_number = jokers
            assert max_joker_number < 0 or max_joker_number >= min_joker_number
            for term in terms:
                if debug_level >= 4:
                    print('  ' * iteration_level, '-', iteration_level, 'LOOK FOR ', term.text, 'jokers', jokers,
                          'debug', debug_level, 'stopper', term.stopper, 'end?', TerminologyTrie.END in current_dict,
                          'from termino?', term.from_terminology, 'dict', len(current_dict))
                    if len(current_dict) == 1:
                        print('  ' * iteration_level, '-', iteration_level, 'current_dict', current_dict)
                #    debug_level = 4

                #
                found_text = None
                # Term lookup
                if min_joker_number <= 0 and term.from_terminology and term.text in current_dict:
                    if debug_level >= 4:
                        print('  ' * iteration_level, '-', iteration_level, '0')
                    found_text = term.text
                # Num ?
                elif min_joker_number <= 0 and self.conf.NUM_JOKER in current_dict and NUM_REGEX.match(
                        term.text) is not None:
                    if debug_level >= 4:
                        print('  ' * iteration_level, '-', iteration_level, '1')
                    found_text = self.conf.NUM_JOKER

                # Found -> yield term or number
                if found_text is not None:
                    if debug_level >= 4:
                        print('  ' * iteration_level, '-', iteration_level, '2. FOUND ', term.text)
                    if matches_met is None:
                        new_matches_met = [term.from_match] if term.from_match is not None else []
                    else:
                        new_matches_met = matches_met + [
                            term.from_match] if term.from_match is not None else matches_met
                    for m in self._match_from(term_list, start, current_end, term.end,
                                              from_index, current_index + 1, current_dict[found_text],
                                              matches_met=new_matches_met,
                                              debug_level=debug_level, iteration_level=iteration_level + 1):
                        if debug_level >= 4:
                            print('  ' * iteration_level, '-', iteration_level, '2.1')
                        yield m
                    if TerminologyTrie.END in current_dict[found_text]:
                        if debug_level >= 4:
                            print('  ' * iteration_level, '-', iteration_level, '2.2')
                        for r in current_dict[found_text][TerminologyTrie.END]:
                            yield r, start, term.end, from_index, current_index, new_matches_met

                # not found a term but a joker raised its limits (meets stopper)
                elif max_joker_number >= 0 and term.stopper and TerminologyTrie.END in current_dict:
                    if debug_level >= 4:
                        print('  ' * iteration_level, '-', iteration_level, '3.1')
                    for r in current_dict[TerminologyTrie.END]:
                        yield r, start, current_end, from_index, current_index - 1, matches_met if matches_met is not None else []
                # not found a term but a joker raised its limits (end of joker)
                elif max_joker_number == 0 and TerminologyTrie.END in current_dict:
                    if debug_level >= 4:
                        print('  ' * iteration_level, '-', iteration_level, '3.2')
                    if matches_met is None:
                        new_matches_met = [term.from_match] if term.from_match is not None else []
                    else:
                        new_matches_met = matches_met + [
                            term.from_match] if term.from_match is not None else matches_met

                    for r in current_dict[TerminologyTrie.END]:
                        yield r, start, term.end, from_index, current_index, new_matches_met
                # not found a term but a joker raised its limits (end of sentence)
                elif max_joker_number > 0 and current_index == len(
                        term_list) - 1 and TerminologyTrie.END in current_dict:
                    if debug_level >= 4:
                        print('  ' * iteration_level, '-', iteration_level, '3.3')
                    if matches_met is None:
                        new_matches_met = [term.from_match] if term.from_match is not None else []
                    else:
                        new_matches_met = matches_met + [
                            term.from_match] if term.from_match is not None else matches_met
                    for r in current_dict[TerminologyTrie.END]:
                        yield r, start, term.end, from_index, current_index, new_matches_met

                # joker expressions (*, ., ...)
                # here we don't care to know whether a matching with current term is possible or not
                # we try a longer match anyway
                if JOKER in current_dict and not term.stopper:
                    if debug_level >= 4:
                        print('  ' * iteration_level, '-', iteration_level, '4')
                    for joker_number in current_dict[JOKER].keys():
                        new_min_joker_number, new_max_joker_number = map(int, joker_number.split('_'))
                        if new_max_joker_number == -1:
                            # *
                            if self.conf.MAX_N <= 0:
                                new_max_joker_number = len(term_list) - current_index
                            else:
                                new_max_joker_number = min(self.conf.MAX_N, len(term_list) - current_index)
                        for m in self._match_from(term_list, start, current_end, current_end,
                                                  from_index, current_index, current_dict[JOKER][joker_number],
                                                  matches_met=matches_met,
                                                  jokers=(new_min_joker_number, new_max_joker_number),
                                                  debug_level=debug_level, iteration_level=iteration_level + 1):
                            yield m

                # currently parsing a joker expression -> minimum number of jokers not reached
                # here we don't care to know whether a matching with current term is possible or not
                # we try a longer match anyway
                # parse all next possible terms
                if max_joker_number > 0 and not term.stopper:
                    if debug_level >= 4:
                        print('  ' * iteration_level, '-', iteration_level, '5 max_joker_number', max_joker_number)
                    if matches_met is None:
                        new_matches_met = [term.from_match] if term.from_match is not None else []
                    else:
                        new_matches_met = matches_met + [
                            term.from_match] if term.from_match is not None else matches_met
                    for m in self._match_from(term_list, start, current_end, term.end,
                                              from_index, current_index + 1, current_dict,
                                              matches_met=new_matches_met,
                                              jokers=(min_joker_number - 1, max_joker_number - 1),
                                              debug_level=debug_level, iteration_level=iteration_level + 1):
                        yield m

                if debug_level >= 4:
                    print('  ' * iteration_level, '-', iteration_level, 'OUT')

    def match_from(self, term_list, from_index=0, debug_level=0):
        '''
        @param term_list list of lists of objects of type Term
        '''
        current_dict = self.root
        terms = term_list[from_index]
        if len(terms):
            start = terms[0].start
            matches = [m for m in self._match_from(term_list, start, start, start,
                                                   from_index, from_index, current_dict,
                                                   debug_level=debug_level)]
            return matches


class TerminologySearcher(object):
    """ The main class to interact with the matcher.
    """

    def __init__(
            self, conf,
            spacy_model_object=None, spacy_model_name='fr_core_news_sm',
            section_splitter_path=None,
            stopwords=None,
            overlapping_criteria='score', threshold=0.7, window=5,
            similarity_name='jaccard', min_match_length=3, rule_parser=None,
            verbose=False):
        """
            This is the main interface through which text can be processed.

        @param spacy_model_object: the spaCy model, already loaded (None if no model).
        @param spacy_model_name: the spaCy model to load (not considered if spacy_model_object is not None)
        Args:
            overlapping_criteria (str, optional):
                    One of "score" or "length". Choose how results are ranked.
                    Choose "score" for best matching score first or "length" for longest match first.. Defaults to 'score'.
            threshold (float, optional): Minimum similarity between strings. Defaults to 0.7.
            window (int, optional): Maximum amount of tokens to consider for matching. Defaults to 5.
            similarity_name (str, optional): One of "dice", "jaccard", "cosine", or "overlap".
                    Similarity measure to be used. Defaults to 'jaccard'.
            min_match_length (int, optional): TODO: ??. Defaults to 3.
            accepted_semtypes (List[str], optional): Set of UMLS semantic types concepts should belong to.
                Semantic types are identified by the letter "T" followed by three numbers
                (e.g., "T131", which identifies the type "Hazardous or Poisonous Substance").
                Defaults to constants.ACCEPTED_SEMTYPES.
            verbose (bool, optional): TODO:??. Defaults to False.
            keep_uppercase (bool, optional): By default QuickUMLS converts all
                    uppercase strings to lowercase. This option disables that
                    functionality, which makes QuickUMLS useful for
                    distinguishing acronyms from normal words. For this the
                    database should be installed without the -L option.
                    Defaults to False.

        Raises:
            ValueError: Raises a ValueError if QuickUMLS was installed for a language that is not currently supported TODO: verify this?
            OSError: Raises an OSError if the required Spacy model was not installed.
        """

        self.verbose = verbose
        self.conf = conf

        valid_criteria = {'length', 'score'}
        err_msg = (
            '"{}" is not a valid overlapping_criteria. Choose '
            'between {}'.format(
                overlapping_criteria, ', '.join(valid_criteria)
            )
        )
        assert overlapping_criteria in valid_criteria, err_msg
        self.overlapping_criteria = overlapping_criteria

        valid_similarities = {'dice', 'jaccard', 'cosine', 'overlap'}
        err_msg = ('"{}" is not a valid similarity name. Choose between '
                   '{}'.format(similarity_name, ', '.join(valid_similarities)))
        assert not (valid_similarities in valid_similarities), err_msg
        self.similarity_name = similarity_name

        self.valid_punct = constants.UNICODE_DASHES

        self.window = window
        self.ngram_length = 3
        self.threshold = threshold
        self.min_match_length = min_match_length
        if stopwords is None:
            self._stopwords = set()
        else:
            self._stopwords = stopwords

        self._info = None

        self.entity_db = DictionaryLookup()
        self.short_term_db = DictionaryLookup()
        self.term_db = DictDatabase(CharacterNgramFeatureExtractor(2))
        self.regexes = {}
        self.entity_to_id = {}
        self.term_to_id = {}
        self.term_searcher = None
        self.stemmer = FrenchStemmer(ignore_stopwords=True)
        # Trie for regular term matching
        self.trie = TerminologyTrie(conf)
        # Tries triggered when meeting special kinds of elements
        #  (used to avoid triggering complex search at all sentences)
        self.specific_tries = dict()

        # multi-expression contraints = entities trigger by presence/absence of other entities
        # Each multi-expression is a list of contraints:
        # [(tuple of OR expressions), (tuple of OR expressions), (NEG expression...)]
        # each of them being conjonctive
        self.multi_expression_contraints = []

        # incremental index for terms created internally
        self.intern_term_index = 0
        # Already registered intern terms
        self.intern_terms = dict()
        # rule parser
        self.rule_parser = rule_parser
        # rules created internally when parsing the terminology
        self.intern_rules = []

        # dictionary (term, set of element ids) of phrases that should never be annotation
        self.exclusion_dict = dict()

        self.jokers = dict()
        for joker in self.conf.WORD_JOKER_UP_TO_5:
            self.jokers[joker] = (0, 5)
        for joker in self.conf.WORD_JOKER_UP_TO_3:
            self.jokers[joker] = (0, 3)
        for joker in self.conf.WORD_JOKER_UP_TO_N:
            self.jokers[joker] = (0, -1)
        for joker in self.conf.WORD_JOKER:
            self.jokers[joker] = (1, 1)
        for joker in self.conf.WORD_OPTIONAL_JOKER:
            self.jokers[joker] = (0, 1)
        self.jokers[self.conf.NUM_JOKER] = 'NUM'

        ##############
        # Init section splitter
        ##############
        if section_splitter_path is not None and self.conf.SECTION_SPLITTER is not None:
            self.section_splitter = SectionSplitter(terminology_path=section_splitter_path,
                                                    mode=self.conf.SECTION_SPLITTER)
        else:
            self.section_splitter = None

        ##############
        # Init spaCy tokenizer
        ##############
        try:
            if spacy_model_object is None:
                self.nlp = spacy.load(spacy_model_name)
            else:
                self.nlp = spacy_model_object
        except OSError:
            msg = (
                'Model "{0}" is not downloaded. Please '
                'run "python -m spacy download {0}" before launching '
            ).format(
                spacy_model_name
            )
            raise OSError(msg)

        ###############
        # Tokenizer customization
        ###############
        # Rule-based sentence splitting (instead of dependency-based)
        if int(spacy.__version__[0]) >= 3:
            if 'sentencizer' not in self.nlp.pipe_names:
                self.nlp.add_pipe('sentencizer', first=True)
            if 'set_custom_boundaries' not in self.nlp.pipe_names:
                print('add pipe')
                self.nlp.add_pipe('set_custom_boundaries')
        else:
            if 'sentencizer' not in self.nlp.pipe_names:
                sentencizer = self.nlp.create_pipe('sentencizer')
                self.nlp.add_pipe(sentencizer, first=True)
            if 'set_custom_boundaries' not in self.nlp.pipe_names:
                self.nlp.add_pipe(set_custom_boundaries)

        # Characters ':', '=', '/' must always be token separator
        # even when not followed by a space
        infixes = self.nlp.Defaults.infixes + self.conf.EXTRA_TOKEN_SEPARATORS
        infix_regex = spacy.util.compile_infix_regex(infixes)
        self.nlp.tokenizer.infix_finditer = infix_regex.finditer
        # NUM_JOKER must be a single token
        if int(spacy.__version__[0]) >= 3:
            self.nlp.tokenizer.add_special_case(self.conf.NUM_JOKER,
                                                [
                                                    {
                                                        symbols.ORTH: self.conf.NUM_JOKER,
                                                        symbols.NORM: self.conf.NUM_JOKER
                                                    }
                                                ])
            # User-defined jokers must be single tokens
            for j in self.jokers.keys():
                self.nlp.tokenizer.add_special_case(j,
                                                    [
                                                        {
                                                            symbols.ORTH: j,
                                                            symbols.NORM: j
                                                        }
                                                    ])
        else:
            self.nlp.tokenizer.add_special_case(self.conf.NUM_JOKER,
                                                [
                                                    {
                                                        symbols.ORTH: self.conf.NUM_JOKER,
                                                        symbols.LEMMA: self.conf.NUM_JOKER,
                                                        symbols.POS: 'NUM'
                                                    }
                                                ])
            # User-defined jokers must be single tokens
            for j in self.jokers.keys():
                self.nlp.tokenizer.add_special_case(j,
                                                    [
                                                        {
                                                            symbols.ORTH: j,
                                                            symbols.LEMMA: j,
                                                            symbols.POS: 'JOK'
                                                        }
                                                    ])

    def load_terminologies(self, in_excel_terminologies, entity_ids, disable_tqdm=False):
        dfs = []

        sheet_ids = []
        for excel_file in tqdm(in_excel_terminologies, disable=disable_tqdm):
            # Open Excel file
            xls = pd.ExcelFile(excel_file)
            # Read all worksheets
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name, header=None,
                                   keep_default_na=False,
                                   names=TERMINO_COLUMN_NAMES,
                                   usecols=[i for i in range(len(TERMINO_COLUMN_NAMES))],
                                   skiprows=[i for i in range(TERMINO_HEADER_HEIGHT)])
                # Do NOT remove empty lines in order to keep line numbers as in the original file
                dfs.append(df)
                sheet_ids.append(excel_file + '_' + sheet_name)
        logging.info('Loading {} terminologies'.format(len(dfs)))

        # Some expressions in the terminology can refer to other entities instead of term
        # (refered as '<ENTITY>')
        # These expressions must be checked after all others
        entity_names_for_second_level_rules = dict()

        total_lines = sum([len(df) for df in dfs])
        pbar = tqdm(total=total_lines, disable=disable_tqdm)
        debug = False
        error = 0

        for file_index, df in enumerate(dfs):
            # if file_index == 0:
            #    continue
            previous_ent_id = ''
            for index, row in df.iterrows():
                ent_id = slugify(row[TERMINO_ID].strip())
                ent_expr = row[EXPRESSION].strip()
                
                pbar.update()

                # skip empty lines
                empty_lines = int(ent_id == '') + int(ent_expr == '')
                if empty_lines == 2:
                    continue
                elif int(ent_expr == ''):
                    logging.error(termino_error_message(sheet_ids[file_index], index, 'Invalid empty cell'))
                    error += 1
                    continue
                
                if type(row[PREANNOTATION]) == float and math.isnan(row[PREANNOTATION]):
                    row[PREANNOTATION] = ''
                if row[PREANNOTATION] not in DISABLED_TERMS:
                    print('empty_lines', empty_lines, ent_id, ent_expr)
                    logging.error(termino_error_message(sheet_ids[file_index], index,
                                                           'Invalid value {} in column Annotation'.format(
                                                               row[PREANNOTATION])))
                    error += 1
                    continue
                
                # Default value for expr_type : ENTITY
                expr_type = row[EXPR_TYPE]
                if type(expr_type) == float and math.isnan(expr_type):
                    expr_type = ''
                if expr_type == '':
                    expr_type = ENTITY
                else:
                    expr_type = expr_type.strip()

                
                # disabled for pre-annotation? (can be used for other rules but won't appear in the final annotation)
                disabled = DISABLED_TERMS[row[PREANNOTATION]]

                # if the entity id is not specified, it's the same as the previous one
                if ent_id == '':
                    ent_id = previous_ent_id
                previous_ent_id = ent_id
                s_ent_id = slugify(ent_id)

                if disabled:
                    continue
                # Check that element id is described in the schema
                if not s_ent_id in entity_ids:
                    logging.error(termino_error_message(sheet_ids[file_index],
                                                           index,
                                                           f'Id {ent_id} is not defined in schema'))
                    error += 1
                    continue

                if not expr_type in EXPRESSION_TYPES:
                    logging.error(termino_error_message(sheet_ids[file_index],
                                                           index,
                                                           f'Invalid expression type {expr_type}'))
                    error += 1
                    continue
                ent_expr = length_preserving_clean_text(ent_expr)

                # ADD to searcher
                entity_name_lists, error_message = self.add(ent_expr, s_ent_id, expr_type, (disabled,), debug=debug)

                if error_message is not None:
                    logging.error(termino_error_message(sheet_ids[file_index], index, error_message))
                    error += 1
                    continue
                # entity_names is the list of entity names present in the expression 
                #   (form <ENTITY>, refering to entity extracted by other rules)
                # These rules will be triggered after all others
                # by replacing the expressions matching entity_names by the corresponding entity name
                # so that this second level rule can match
                #if len(entity_name_lists):
                #    print('entity_name_lists', entity_name_lists)
                for entity_names in entity_name_lists:
                    if type(entity_names) == str:
                        entity_names = [entity_names]
                    for s_entity_name in entity_names:
                        if s_entity_name not in entity_ids and not s_entity_name.startswith(INTERN_TERM_PREFIX):
                            logging.error(termino_error_message(sheet_ids[file_index],
                                                                index,
                                                                f'Id {s_entity_name} is not defined in schema'))
                            error += 1
                            continue
                        entities_to_change = entity_names_for_second_level_rules.get(s_entity_name, set())
                        entities_to_change.add(tuple(sorted(entity_names)))
                        entity_names_for_second_level_rules[s_entity_name] = entities_to_change
                # pbar.refresh()
                
        # if at least one error has been detected, raise ValueError
        # so that the user can't continue without seeing the error
        if error > 0:
            raise ValueError(f'{error} line(s) in the terminology files contained an error, '
                             'see error messages above')
                
        pbar.close()
        return entity_names_for_second_level_rules

    def get_token_regex(self, joker_class):
        """
        [^\S\r\n] => spaces but not newlines (\h does not exist in python)
        [^ ,.()\n] => word character (\w is no reliable)
        [^ \t\n\r\f\v]
        """
        if type(joker_class) == int:
            if joker_class <= 0:
                return '[^\s,.()]+(?:\s+[^\s,.()]+)*'
            elif joker_class == 1:
                return '[^\s,.()]+'
            else:
                return '[^\s,.()]+(?:\s+[^\s,.()]+){1,' + str(joker_class) + '}'
        elif joker_class == 'NUM':
            return self.conf.NUM_REGEX
        else:
            raise ValueError(joker_class)

    def search_token(self, token, regex_only=False):
        if self.term_searcher is None:
            self.term_searcher = Searcher(self.term_db, CosineMeasure())

        # First, look for exact match (acronyms or short word and element tagged as entities)
        results = self.entity_db.search(token)
        term_results = []
        if len(results) == 0:
            # if not found, look for approximate match
            if len(token) > APPROX_MATCHING_LENGTH_MIN:
                results = self.term_searcher.search(slugify(token).lower(), self.threshold)
            # if the term is short, look into short_term_db (no approximate matching but case-insensitive)
            else:
                results = self.short_term_db.search(token.lower())

            if not regex_only:
                term_results = [(r, self.term_to_id[r]) for r in results]  # if len(r) > APPROX_MATCHING_LENGTH_MIN]
            regex_results = []
        else:
            if not regex_only:
                term_results = [(r, self.entity_to_id[r]) for r in results]
            regex_results = []

        return term_results, regex_results

    def add_term_or_entity(self, text, element_id, element_type):
        # Add as entity (no approximate search allowed)
        if text.isupper() or element_type == ENTITY:
            self.entity_db.add(text)
            types = self.entity_to_id.get(text, set())
            types.add(element_id)
            self.entity_to_id[text] = types
        # Add as term (approximate search allowed)
        else:
            # short "Terms" are case-insensitive but we restrict to strict matching
            if element_type == TERM and len(text) <= APPROX_MATCHING_LENGTH_MIN:
                text = text.lower()
                self.short_term_db.add(text)
            # long term allow approximate matching
            else:
                text = slugify(text)
                self.term_db.add(text)

            types = self.term_to_id.get(text, set())
            types.add(element_id)
            self.term_to_id[text] = types
        return text

    def add_regex(self, regex, element_id):
        """
        @return error message (None or str)
        """
        try:
            # edit regex for <NUM> special token
            # + other basic checks
            regex = regex.replace(self.conf.NUM_JOKER, self.conf.NUM_REGEX)
            c_regex = re.compile(regex)
        except Exception as err:
            return str(err)
        if c_regex in self.regexes:
            return "Duplicate regex {} (skip)".format(regex)
        self.regexes[c_regex] = element_id
        return None

    def add_tokens(self, tokens, element_id, element_type, element_features,
                   debug=False):
        running_terms = []
        entity_name = None
        all_entity_names = []       # all list of second-level entities 
        running_entity_names = []   # the main second-level entities in this expression
        in_entity_name = False

        for i, token in enumerate(tokens):
            if token in ('', ' ', '\n'):
                continue
            # Entity token (represents an entity extracted by a previous rule)
            # must be of the from "<ENTITY>"
            # Start of the entity is '<'
            # End of the entity is '>'
            # the entity name is in-between
            # Entity name (between '<' and '>')
            if in_entity_name:
                if token == '>':
                    in_entity_name = False
                    s_entity_name = slugify('_'.join(entity_name))
                    
                    running_terms.append(s_entity_name)
                    running_entity_names.append(s_entity_name)
                    if debug:
                        print('found entity name', s_entity_name)
                    entity_name = None
                else:
                    entity_name.append(token)
            # Start of an entity name
            elif token == '<':
                if any([t == '>' for t in tokens[i + 1:]]):
                    in_entity_name = True
                    entity_name = []
            # Regular token
            elif token not in self.jokers:
                slug_text = slugify(token)
                cleaned_token = self.clean_token(token)
                running_terms.append(slug_text)
                self.add_term_or_entity(cleaned_token, slug_text, element_type)
            # NUM token
            elif token == self.conf.NUM_JOKER:
                running_terms.append(token)
            # joker token
            elif token in self.jokers:
                # If the joker is not the first token, just append it
                if i > 0:
                    running_terms.append(JOKER)
                    running_terms.append('{}_{}'.format(*self.jokers[token]))
                    # If the joker is the first token, this needs a special treatment
                # otherwise this will lead to poor performance (all n-grams in the corpus
                # will match an expression starting with a joker...)
                # In this case we register the tokens following this one as an internal
                # element, and create a rule that will be triggered only when this
                # internal expression is met
                else:
                    tokens_after_joker = tokens[i + 1:]
                    tokens_str = ' '.join(tokens_after_joker)
                    # If the operand has already been registered as an intern term
                    # get its element_id and re-use it
                    if tokens_str in self.intern_terms:
                        new_element_id = self.intern_terms[tokens_str]
                    # Otherwise, create a new intern term
                    else:
                        new_element_id = INTERN_TERM_PREFIX + str(self.intern_term_index)
                        new_entity_names, error_message = self.add_tokens(tokens_after_joker, new_element_id,
                                                                          element_type, (True,)) # 'True' stands for "disable annotation in Brat"
                        if error_message is not None:
                            return [], error_message
                        else:
                            #print(f'""""""""{new_entity_names}"""""""""""""')
                            all_entity_names.extend(new_entity_names)
                            all_entity_names.append([new_element_id])
                            #running_entity_names.append(new_element_id)
                        #print('entity_names', all_entity_names)
                        #print('new_entity_names', new_entity_names)
                        self.intern_term_index += 1
                        self.intern_terms[tokens_str] = new_element_id
                    new_expression = token + ' <{}>'.format(new_element_id)
                    if self.conf.debug_level == 3:
                        logging.debug('Add new internal expression {} for an expression starting by a joker'.format(
                            new_expression))
                    # The new expression is the joker + the new_element_id
                    terms_for_intern_term = [JOKER,
                                             '{}_{}'.format(*self.jokers[token]),
                                             new_element_id]
                    #print('%%%%%%%%%%ù', new_entity_names)
                    if debug:
                        print('add entity_names to specific tries')
                    str_sorted_entity_name = new_element_id #'_'.join(sorted(new_entity_name_list))
                    spec_trie = self.specific_tries.get(str_sorted_entity_name, TerminologyTrie(self.conf))
                    spec_trie.add(terms_for_intern_term, (element_id, *element_features))
                    self.specific_tries[str_sorted_entity_name] = spec_trie
                    if debug:
                        print('self.specific_tries', self.specific_tries)

                    break
            else:
                raise NotImplementedError()

        if len(running_entity_names):
            all_entity_names.append(running_entity_names)
                
        if len(all_entity_names) == 0:
            self.trie.add(running_terms, (element_id, *element_features))
        else:
            if debug:
                print('add entity_names to specific tries')
            str_sorted_entity_name = '_'.join(sorted(running_entity_names))
            #print("FINNNNNNNNN", str_sorted_entity_name)
            #print('')
            spec_trie = self.specific_tries.get(str_sorted_entity_name, TerminologyTrie(self.conf))
            spec_trie.add(running_terms, (element_id, *element_features))
            self.specific_tries[str_sorted_entity_name] = spec_trie
            if debug:
                print('self.specific_tries', self.specific_tries)
        if debug:
            print('return OK', all_entity_names)
        #print('all_entity_names', all_entity_names)
        return all_entity_names, None

    def add(self, text, element_id, element_type, element_features, debug=False):
        '''
        @param text: the string representation of the mention
        @param element_id : the entity type
        @param element_type : the type of the expression (term, entity, regex...)
        @param element_features : tuple containing features associated to the element (will be returned with the element_id when searched)

        @returns the entity names present in the expression ('<ENTITY>')
        '''

        # Reset term searcher (outdated when we add new elements)
        self.term_searcher = None

        if element_type not in EXPRESSION_TYPES:
            raise ValueError(element_type)

        # add to exclusion list
        if element_type == EXCLUSION:
            element_id_set = self.exclusion_dict.get(text, set())
            element_id_set.add(element_id)
            self.exclusion_dict[text] = element_id_set
            return [], None

        # add REGEX
        if element_type == REGEX:
            error_message = self.add_regex(text, element_id)
            # no entity names for second-level parsing in regexes
            return [], error_message

        # add TERM or ENTITY

        # 1. Split by SAME_SENTENCE_OPERATOR : each operand around this operand
        #     will be a separate term
        operands = text.split(self.conf.SAME_SENTENCE_OPERATOR)

        if len(operands) > 1:
            # entity names present in the expression
            #  (expressed between '<' and '>', except for <NUM>)
            entity_names = []
            if self.rule_parser is None:
                return [], "Expression with two operands but no rule parser available"
            new_elements = []
            for op in operands:
                op = op.strip()
                # If the operand has already been registered as an intern term
                # get its element_id and re-use it
                if op in self.intern_terms:
                    new_element_id = self.intern_terms[op]
                # Otherwise, create a new intern term
                else:
                    new_element_id = INTERN_TERM_PREFIX + str(self.intern_term_index)
                    # 'True' stands for "disable annotation in Brat"
                    new_entity_names, error_message = self.add(op, new_element_id, element_type, (True,))
                    if error_message is not None:
                        return entity_names, error_message
                    else:
                        entity_names.extend(new_entity_names)
                        self.intern_term_index += 1
                        self.intern_terms[op] = new_element_id
                new_elements.append(new_element_id)

                # Create new rule with high id (to be ranked after the used-defined rules and 'add' mode)
            new_rule = (10000, self.conf.RULE_ADD,
                        element_id, self.rule_parser.parse(protect_rule(' AND '.join(new_elements), self.conf)))
            if self.conf.debug_level == 3:
                logging.debug('Add new internal rule {}'.format(new_rule))
            self.intern_rules.append(new_rule)
            return entity_names, None

        # 2. Here, there is only one operand in the term (i.e. single term without SAME_SENTENCE_OPERATOR)
        else:
            # Add space around special token separators
            for sep in self.conf.EXTRA_TOKEN_SEPARATORS:
                text = text.replace(sep, ' ' + sep + ' ')
                
            # "Protect" the <NUM> joker (but then word<NUM> with no space is not possible)
            # TODO work on this issue
            text = text.replace(self.conf.NUM_JOKER, ' ' + self.conf.NUM_JOKER + ' ')

            tokens = [t.text for t in self.nlp('{}'.format(text))]
            
            # Fix: when the text ends with "d'" or "l'" for example
            # Spacy tokenizes as "d" + "'" (while it does it correctly when there is a word after the "'")
            # The following line fixes that
            if tokens[-1] == "'":
                tokens[-2] = tokens[-2] + "'"
                tokens = tokens[:-1]

            if debug:
                print('tokens', tokens)
            entity_names, error_message = self.add_tokens(tokens, element_id, element_type, element_features,
                                                          debug=debug)
            # return list of second-level entity names,  error message
            #if len(entity_names) > 1:   ## TODO REMOVED BUT MIGHT BREAK EVERYTHING
            #    entity_names = [entity_names]
            return entity_names, error_message

    def get_info(self):
        """Computes a summary of the matcher options.

        Returns:
            Dict: Dictionary containing information on the QuicUMLS instance.
        """
        return self.info

    @property
    def info(self):
        """Computes a summary of the matcher options.

        Returns:
            Dict: Dictionary containing information on the QuicUMLS instance.
        """
        # useful for caching of respnses

        if self._info is None:
            self._info = {
                'threshold': self.threshold,
                'similarity_name': self.similarity_name,
                'window': self.window,
                'ngram_length': self.ngram_length,
                'min_match_length': self.min_match_length
                # 'accepted_semtypes': sorted(self.accepted_semtypes),
                # 'negations': sorted(self.negations),
                # 'valid_punct': sorted(self.valid_punct)
            }
        return self._info

    def _is_valid_token(self, tok):
        return not (
            # tok.is_punct or tok.is_space or
                tok.pos_ == 'PUNCT' or tok.is_space or
                tok.pos_ == 'ADP' or tok.pos_ == 'DET' or tok.pos_ == 'CONJ'
        )

    def _is_stop_term(self, tok):
        return tok.text in self._stopwords

    def _is_valid_end_token(self, tok):
        return not (
            # tok.is_punct or tok.is_space or self._is_stop_term(tok) or
                tok.pos_ == 'PUNCT' or tok.is_space or self._is_stop_term(tok) or
                tok.pos_ == 'ADP' or tok.pos_ == 'DET' or tok.pos_ == 'CONJ'
        )

    def _is_valid_middle_token(self, tok):
        return (
            # not(tok.is_punct or tok.is_space) or
                not (tok.pos_ == 'PUNCT' or tok.is_space) or
                tok.is_bracket or
                tok.text in self.valid_punct
        )

    # def _is_ok_semtype(self, target_semtypes):
    #    if self.accepted_semtypes is None:
    #        ok = True
    #    else:
    #        ok = any(sem in self.accepted_semtypes for sem in target_semtypes)
    #    return ok

    def _is_longer_than_min(self, span):
        return (span.end_char - span.start_char) >= self.min_match_length

    def stem_token(self, str_token):
        if str_token.isupper():
            return self.stemmer.stem(str_token).upper()
        else:
            return self.stemmer.stem(str_token)

    def clean_token(self, token):
        if type(token) == tokens.token.Token:
            token_text = token.text
        elif type(token) == str:
            token_text = token
        else:
            raise TypeError()

        if len(token_text) <= APPROX_MATCHING_LENGTH_MIN:
            return token_text
        else:
            return self.stem_token(token_text)

    def _keep_longest_matches(self, values):
        (start1, end1, r_id1) = values[0]
        keep = []
        for i in range(1, len(values)):
            (start2, end2, r_id2) = values[i]
            # same regex id ?
            if r_id1 == r_id2:
                # overlap ?
                if end1 > start2:
                    # keep longest match
                    if end1 - start1 < end2 - start2:
                        # keep 2nd
                        return self._keep_longest_matches(values[1:])
                    else:
                        # keep first
                        # -> do nothing (won't keep the 2nd on next iteration)
                        ...
                # no overlap
                else:
                    keep.append((start2, end2, r_id2))
            # different regex id
            else:
                keep.append((start2, end2, r_id2))
        if len(keep):
            return [values[0]] + self._keep_longest_matches(keep)
        else:
            return [values[0]]

    @staticmethod
    def _select_score(match):
        return (match[0]['similarity'], (match[0]['end'] - match[0]['start']))

    @staticmethod
    def _select_longest(match):
        return ((match[0]['end'] - match[0]['start']), match[0]['similarity'])

    def _select_terms(self, matches_by_id, is_a_hierarchy):
        match_list = [m for _, m in matches_by_id.items()]
        assert all([len(m) == 1 for m in match_list])

        sort_func = (
            self._select_longest if self.overlapping_criteria == 'length'
            else self._select_score
        )

        match_list = sorted(match_list, key=sort_func, reverse=True)

        intervals = toolbox.Intervals(is_a_hierarchy)
        final_matches_subset = dict()

        for match in match_list:
            match_interval = (match[0]['start'], match[0]['end'], match[0]['type'])
            if match_interval not in intervals:
                final_matches_subset[match[0]['id']] = match
                intervals.append(match_interval)

        return final_matches_subset

    def _make_token_sequences(self, parsed, sent_start_char):
        for i in range(len(parsed)):
            for j in xrange(
                    i + 1, min(i + self.window, len(parsed)) + 1):
                span = parsed[i:j]

                if not self._is_longer_than_min(span):
                    continue

                yield (span.start_char - sent_start_char, span.end_char - sent_start_char, span.text)

    def _print_verbose_status(self, parsed, matches):
        if not self.verbose:
            return False

        print(
            '[{}] {:,} extracted from {:,} tokens'.format(
                datetime.datetime.now().isoformat(),
                sum(len(match_group) for match_group in matches),
                len(parsed)
            ),
            file=sys.stderr
        )
        return True

    def get_match_union(self, matches):
        min_offset = 1000000
        max_offset = 0
        for match in matches:
            if match['start'] < min_offset:
                min_offset = match['start']
            if match['end'] > max_offset:
                max_offset = match['end']
        return min_offset, max_offset

    def get_match_hash(self, start, end, elem_id):
        return str(start) + '-' + str(end) + '-' + elem_id

    def transitive_closure(self, matches_by_id):
        changed = False
        for match_id, match in matches_by_id.items():
            if 'related' in match:
                related = set(match['related'])
                for r in related:
                    related_match = matches_by_id[r]
                    if 'related' in related_match:
                        to_add = [m_id for m_id in related_match['related'] if m_id not in related]
                        if len(to_add):
                            changed = True
                            related.update(to_add)
                match['related'] = list(related)
        if changed:
            return self.transitive_closure(matches_by_id)
        else:
            return matches_by_id

    def follow_INTERN_match(self, match_id, sent_matches_by_id, visited=None):
        all_matches = set()

        if INTERN_TERM_PREFIX in match_id:
            match = sent_matches_by_id[match_id]
            if visited is None:
                visited = {match_id}
            else:
                visited.add(match_id)
            related = match['related']
            for r in related:
                if r not in visited:
                    all_matches.update(self.follow_INTERN_match(r, sent_matches_by_id, visited=visited))
        else:
            return {match_id}
        return all_matches

    def relate_them_all(self, matches, sent_matches_by_id):
        # Remove matches concerning SECTIONs
        filtered_matches = [match_id for match_id in matches if self.conf.SECTION_PREFIX not in match_id]
        if len(filtered_matches) > 1:
            # Build all one-vs-all relations and add them to current 'related' attribute
            for i in range(len(filtered_matches)):
                match_list = sent_matches_by_id[filtered_matches[i]]
                for match in match_list:
                    related = set(match.get('related', []))
                    for j, m_id in enumerate(filtered_matches):
                        if i != j:
                            related.update(self.follow_INTERN_match(m_id, sent_matches_by_id))
                    match['related'] = list(related)
                sent_matches_by_id[filtered_matches[i]] = match_list
        return sent_matches_by_id

    
    def apply_rules(self, sent_index, sent_matches_by_id, sent_matches_by_ent, rule_dict,
                    sent_text, sent_start,
                    is_a_hierarchy,
                    add_intern_rules=False, debug_level=0):
        '''
        Rule matching
        '''
        # Get all rules concerned by the matched entities
        candidate_rules = set()
        for s_match_id in sent_matches_by_ent:
            candidate_rules.update(rule_dict.get(s_match_id, []))

        deleted_ids = set()
        # sort rules by their ids (order of rule triggering is important)
        # if not first pass, add internal rules
        if not add_intern_rules:
            ordered_rules = sorted(list(candidate_rules), key=lambda r: r[0])
        else:
            ordered_rules = sorted(list(candidate_rules) + self.intern_rules, key=lambda r: r[0])

        if debug_level >= 4:
            print('\nLOOK FOR RULES', candidate_rules)
        # ordered_rules can be updated in the loop, that's why we parse and pop
        # until the list is empty instead of parsing with for
        while len(ordered_rules) > 0:
            (rule_id, action, s_ent_id, rule) = ordered_rules.pop(0)
            # entity id is already slugified in the rule store
            checked, found_matches = check_rule(rule, sent_matches_by_ent, sent_matches_by_id, self.conf)

            if debug_level >= 4:
                print('RULE', rule_id, rule)
                print('sent_matches_by_ent', sent_matches_by_ent)
                print('found_matches', found_matches)
            # 1. If the rule is checked:
            #   1.1 if the rule is a REMOVE rule, then remove the entity
            #   1.2 if the rule is a ADD rule, add the new entity and link
            #       it to the entities in the rule
            #   1.3 if the rule is a LINK rule, add the link in the related
            #       attribute of the entity
            # 2. If the rule is not checked:
            #   2.1 if the rule is a REMOVE rule, then link all the
            #       entities in the rule together
            #   2.2 if the rule is a ADD, do nothing
            # 1.
            if checked:
                if debug_level >= 4:
                    print('OK', rule, action, s_ent_id)
                # 1.1
                if action == self.conf.RULE_REMOVE:
                    
                    # get potential match ids
                    # if the s_ent_is is "any entity", get them all
                    if s_ent_id == ANY_ENTITY_TOKEN:
                        match_ids = list({val for subset in sent_matches_by_ent.values() for val in subset})
                    # else get the entities corresponding to the entity type
                    elif s_ent_id in sent_matches_by_ent:
                        match_ids = sent_matches_by_ent[s_ent_id]
                    else:
                        match_ids = []
                    # Delete all match ids that are in found matches.
                    for match_id in match_ids:
                        if match_id in found_matches and match_id in sent_matches_by_id:
                            if debug_level >= 4:
                                print('del ', match_id)
                            del sent_matches_by_id[match_id]
                            deleted_ids.add(match_id)
                # 1.2
                elif action == self.conf.RULE_ADD:
                    start, end = self.get_match_union([sent_matches_by_id[i][0] for i in found_matches])
                    # Enabled BRAT annotation only if the new entity is known by the schema
                    if s_ent_id not in is_a_hierarchy:
                        annotation_disabled = True
                    else:
                        annotation_disabled = False

                    match_hash = self.get_match_hash(start, end, s_ent_id)
                    match = {
                        'start': start,
                        'end': end,
                        'start_index': None,
                        'end_index': None,
                        'sent_index': sent_index,
                        'ngram': sent_text[start - sent_start:end - sent_start],
                        'type': {s_ent_id},
                        'annotation_disabled': annotation_disabled,
                        'similarity': 1.0,
                        'id': match_hash,
                        'related': found_matches,
                        'origin': 'rule'
                    }
                    sent_matches_by_id[match_hash] = [match]
                    sent_match_by_ent_id = sent_matches_by_ent.get(s_ent_id, set())
                    sent_match_by_ent_id.add(match_hash)
                    sent_matches_by_ent[s_ent_id] = sent_match_by_ent_id
                    new_rules = rule_dict.get(s_ent_id, [])
                    if len(new_rules) > 0:
                        ordered_rules.extend(new_rules)
                        # resort based on the new rules
                        ordered_rules = sorted(ordered_rules, key=lambda r: r[0])
                # 1.3
                elif action == self.conf.RULE_LINK:

                    # get potential match ids
                    # if the s_ent_is is "any entity", get them all
                    if s_ent_id == ANY_ENTITY_TOKEN:
                        match_ids = list({val for subset in sent_matches_by_ent.values() for val in subset})
                    # else get the entities corresponding to the entity type
                    elif s_ent_id in sent_matches_by_ent:
                        match_ids = sent_matches_by_ent[s_ent_id]
                    else:
                        match_ids = []
                    # Add links
                    for match_id in match_ids:
                        if match_id in found_matches and match_id in sent_matches_by_id:
                            if debug_level >= 4:
                                print('link ', match_id)
                            matches = sent_matches_by_id[match_id]

                            for m in matches:
                                m['related'] = list(set(m['related'] + [m_id
                                                                        for m_id in found_matches
                                                                        if m_id != match_id]))


                else:
                    raise ValueError()
            # 2 (not checked)
            else:
                # 2.1
                if action == self.conf.RULE_REMOVE and len(found_matches) > 0:
                    sent_matches_by_id = self.relate_them_all(found_matches, sent_matches_by_id)
                    # if any([i for i in found_matches if 'SECTION' not in i]):
                    #    print('----------')
                    #    for found_match_id in found_matches:
                    #        found_match = sent_matches_by_id[found_match_id]
                    #        if 'SECTION' not in found_match_id:
                    #            print('found match', found_match_id, found_match)
        return sent_matches_by_id, sent_matches_by_ent, deleted_ids

    def match_in_sentence(self, sent_index, matched_unigrams, sent_text, sent_start, sent_end,
                          rule_dict, doc_text,
                          is_a_hierarchy=None, sections_df_row=None, regexes_df=None,
                          second_level_rules=None, sent_matches_by_id=None, sent_matches_by_ent=None,
                          seed_term=None, 
                          best_match=True,
                          recursive_call=0,
                          debug_level=0):
        #if recursive_call > 1:
        #    print('call ', recursive_call)
        if is_a_hierarchy is None:
            is_a_hierarchy = {}
        if second_level_rules is None:
            second_level_rules = {}
        if sent_matches_by_id is None:
            sent_matches_by_id = {}
        if sent_matches_by_ent is None:
            sent_matches_by_ent = {}
        if debug_level >= 4:
            print('unigrams', [[str(t) for t in terms] for terms in matched_unigrams])
            print('sent_matches_by_id', sent_matches_by_id)

        candidate_second_level_expressions = set()

        #########
        # Regex matching
        # (only first pass)
        #########
        # Project the information about regexes on this specific sentence
        if seed_term is None:  # seed_term is None means that this is the first pass, not a recursive call
            sent_regex_df = regexes_df[(regexes_df['end'] > sent_start) & (regexes_df['start'] < sent_end)]
            for _, sent_regex_row in sent_regex_df.iterrows():
                start = sent_regex_row['start']
                end = sent_regex_row['end']
                span = doc_text[start:end]
                s_match_id = sent_regex_row['match_type']
                
                #start = max(sent_start, sent_regex_row['start'])
                #end = min(sent_end, sent_regex_row['end'])
                # match_id is already slugified in the regex store
                assert s_match_id == slugify(s_match_id)
                
                # strip spaces around the matching (brat doesn't like spaces)
                ls_span = span.lstrip()
                if span != ls_span:
                    start += (len(span) - len(ls_span))
                s_span = ls_span.rstrip()
                if s_span != ls_span:
                    end -= (len(ls_span) - len(s_span))
                span = s_span
                
                match_hash = self.get_match_hash(start, end, s_match_id)

                match = {
                    'start': start,
                    'end': end,
                    'start_index': None,
                    'end_index': None,
                    'sent_index': sent_index,
                    'ngram': span,
                    'type': {s_match_id},
                    'annotation_disabled': False,
                    'similarity': 1.0,
                    'id': match_hash,
                    'related': [],
                    'origin':'regex'
                }
                
                # Additional information for annotation overlapping two sentences
                # (can happen when created by regexes)
                if sent_start > start:
                    match['extend_to_left_sentence'] = True
                if sent_end < end:
                    match['extend_to_right_sentence'] = True

                sent_matches_by_id[match_hash] = [match]

                # Update dictionary storing matches by entity ids
                for h_match_id in [s_match_id] + is_a_hierarchy.get(s_match_id, []):

                    sent_match_by_ent_id = sent_matches_by_ent.get(h_match_id, set())
                    sent_match_by_ent_id.add(match_hash)
                    sent_matches_by_ent[h_match_id] = sent_match_by_ent_id

                    if debug_level >= 4:
                        print('   regex match: ', s_match_id)

                    # look for candidate 2nd level expressions
                    candidate_second_level_expressions.update(second_level_rules.get(h_match_id, []))

                # look for candidate rules
                # assert not (match_id != s_match_id and match_id in rule_dict)
                # candidate_rules.update(rule_dict.get(s_match_id, []))

        if debug_level >= 4:
            print('sent_matches_by_ent after Regex match', sent_matches_by_ent)

        #########
        # Section matching
        # (only first pass)
        #########
        # Project the information about sections on this specific sentence
        if seed_term is None and sections_df_row is not None:  # seed_term is None means that this is the first pass, not a recursive call
            # sent_sections_df = sections_df[(sections_df['end'] >= sent_start) & (sections_df['start'] <= sent_end)]
            # for _, sent_section_row in sent_sections_df.iterrows():
            start = max(sent_start, sections_df_row['start'])
            end = min(sent_end, sections_df_row['end'])
            match_id = self.conf.SECTION_PREFIX + sections_df_row['match_type']
            if sections_df_row['match_type'] not in self.conf.SECTION_TYPES:
                raise ValueError("Section {} has been found by the section splitter but "
                                 "is not listed as a regular section header "
                                 "(see configuration.py)".format(sections_df_row['match_type']))
            # BRAT-ONLY SECTION ANNOTATION
            # Disable annotation if not the first one of this section
            # -> keep only in Brat the first character of each section
            if sections_df_row['start'] < sent_start:
                disable_annotation = True
            else:
                disable_annotation = False
            s_match_id = slugify(match_id)
            brat_end = start + 1  # start+1 is used to annotate the sections in Brat. To change to `end` for real span
            match_hash = self.get_match_hash(start, brat_end, s_match_id)
            match = {
                'start': start,
                'end': brat_end,  
                'start_index': None,
                'end_index': None,
                'sent_index': sent_index,
                'ngram': sent_text[start - sent_start:end - sent_start],
                'type': {s_match_id},
                'annotation_disabled': disable_annotation,
                'similarity': 1.0,
                'id': match_hash,
                'related': [],
                'out': 'brat_only',
                'origin': 'section_matching'
            }
            sent_matches_by_id[match_hash] = [match]

            for h_match_id in [s_match_id] + is_a_hierarchy.get(s_match_id, []):
                if debug_level >= 4:
                    print('   --> add ', h_match_id)
                # Update dictionary storing matches by entity ids
                sent_match_by_ent_id = sent_matches_by_ent.get(h_match_id, set())
                sent_match_by_ent_id.add(match_hash)
                sent_matches_by_ent[h_match_id] = sent_match_by_ent_id

                # look for candidate 2nd level expressions
                candidate_second_level_expressions.update(second_level_rules.get(h_match_id, []))

        if debug_level >= 4:
            print('sent_matches_by_ent after Section match', sent_matches_by_ent)

        #########
        # Terminology matching
        #########
        # We first use the main trie to search for matching expressions (first pass)
        #    in this is a recursive call, use the specific trie for the element that triggered the recursive cal
        if seed_term is None:
            trie_to_use = self.trie
        else:
            trie_to_use = self.specific_tries[seed_term]

        # Match all sequences starting by each token
        # For each token in match_unigrams
        for i in range(len(matched_unigrams)):
            matches = trie_to_use.match_from(matched_unigrams, from_index=i, debug_level=debug_level)

            if matches is not None and len(matches):
                # each match is:
                # - match_value(a tuple containing the type of the matched entity and other features)
                # - start, end (offsets)
                # - start_index, end_index (token positions)
                # - matches_met: the list of matches from previous iterations that have been met during this match
                for (match_value, start, end, start_index, end_index, matches_met) in matches:
                    s_match_id = match_value[0]  # type of the entity
                    # match_id is already slugified in the searcher
                    assert s_match_id == slugify(s_match_id)

                    preannotation_disabled = match_value[1]  # will the entity appear in the final annotations
                    matched_text = sent_text[start - sent_start:end - sent_start]

                    # check exclusion list
                    if matched_text in self.exclusion_dict:
                        exclude = False
                        for e_id in self.exclusion_dict[matched_text]:
                            if e_id == s_match_id or e_id in is_a_hierarchy.get(s_match_id, []):
                                exclude = True
                                break
                        if exclude:
                            continue

                    # Build match
                    match_hash = self.get_match_hash(start, end, s_match_id)
                    match = {
                        'start': start,
                        'end': end,
                        'start_index': start_index,
                        'end_index': end_index,
                        'sent_index': sent_index,
                        'ngram': matched_text,
                        'type': {s_match_id},
                        'annotation_disabled': preannotation_disabled,
                        'similarity': 1.0,
                        'id': match_hash,
                        'related': matches_met,
                        'origin': 'terminology'
                    }

                    if debug_level >= 4:
                        print('   match ', match_hash, start, '-', end, '-', s_match_id, '-', matched_text)
                    sent_matches_by_id[match_hash] = [match]
                    # Create a match for the entity id and its ancestors
                    for h_match_id in [s_match_id] + is_a_hierarchy.get(s_match_id, []):
                        if debug_level >= 4:
                            print('   --> add ', h_match_id)
                        # Update dictionary storing matches by entity ids
                        sent_match_by_ent_id = sent_matches_by_ent.get(h_match_id, set())
                        sent_match_by_ent_id.add(match_hash)
                        sent_matches_by_ent[h_match_id] = sent_match_by_ent_id

                        # if h_match_id is not None and "negation" in h_match_id.lower():
                        #    print('s_match_id', h_match_id)
                        #    print('second_level_rules.get(h_match_id, [])', second_level_rules.get(h_match_id, []))

                        # print('\n\nextend rule to match ', s_match_id, rule_dict.get(h_match_id, []), '\n\n')
                        # look for candidate 2nd level expressions
                        candidate_second_level_expressions.update(second_level_rules.get(h_match_id, []))

                        # look for candidate rules
                        # candidate_rules.update(rule_dict.get(s_match_id, []))

        if debug_level >= 4:
            print('sent_matches_by_id after term match', sent_matches_by_id)
            print('sent_matches_by_ent after term match\n', sent_matches_by_ent)

        assert not any([len(v) != 1 for _, v in sent_matches_by_id.items()])
        #print('RULE DICT')
        #print(rule_dict)
        #########
        # Rule matching
        #########
        all_deleted_ids = set()
        sent_matches_by_id, sent_matches_by_ent, deleted_ids = self.apply_rules(sent_index,
                                                                   sent_matches_by_id, sent_matches_by_ent, rule_dict,
                                                                   sent_text, sent_start, is_a_hierarchy,
                                                                   add_intern_rules=seed_term is not None,
                                                                   debug_level=debug_level)

        if debug_level >= 4:
            print('\nsent_matches_by_id after rule matching, before 2nd pass\n\n', sent_matches_by_id)
        all_deleted_ids.update(deleted_ids)
        #########
        # 2nd pass with second_level_expressions
        #########
        # Keep best (avoid overlap)
        sent_matches_by_id = self._select_terms(sent_matches_by_id, is_a_hierarchy)
        #if seed_term is None:  # seed_term is None means that this is the first pass, not a recursive call
        # Try to trigger expressions that contained entity names ('<ENTITY>')
        if debug_level >= 4:
            print('\n{} candidate_second_level_expressions\n'.format(len(candidate_second_level_expressions)),
                  candidate_second_level_expressions)
            print('self.intern_rules', self.intern_rules)
        # print('\n')
        for candidate_second_level_expression in candidate_second_level_expressions:
            # candidate_second_level_expression is a set of entity ids that are together in at least one
            # second-level expression. It is then necessary to replace, in the original sentence, the matched text
            # by the name of the entity, in order to loop for the second-level expression
            #
            # TODO Implement for more than 1 entity name in a second-level expression
            # if len(candidate_second_level_expression) > 1:
            #    print('sent_matches_by_ent', sent_matches_by_ent)
            #    raise NotImplementedError('2nd level expressions are not implemented for more than one entity name in the expression ({})'.format(candidate_second_level_expression))
            if debug_level >= 4:
                print('  2nd-level expression : ', candidate_second_level_expression)
                print('++++++++++++++++++++++++')
                print('sent_matches_by_ent', sent_matches_by_ent)
                print('++++++++++++++++++++++++')

            # s_ent_id = candidate_second_level_expression[0]
            # entity id is already slugified
            # assert s_ent_id == slugify(s_ent_id), candidate_second_level_expression

            # check that the entities exist (might have been removed by the rules at previous step)
            if all([s_ent_id in sent_matches_by_ent for s_ent_id in candidate_second_level_expression]):
                if debug_level == 4:
                    print('GO')
                    print('sent_matches_by_ent', sent_matches_by_ent)
                    print('sent_matches_by_id', sent_matches_by_id)

                # Get all entity matches (most frequently, one per entity),
                # i.e parse all entities and get the matches in the sentence corresponding to this entity id.
                cancel = False
                all_ent_matches = []
                for s_ent_id in candidate_second_level_expression:
                    assert not any([len(sent_matches_by_id[i]) != 1 for i in sent_matches_by_ent[s_ent_id] if
                                    i in sent_matches_by_id])
                    # can be empty if the entity has been removed during the rule triggering
                    # if empty, do not keep going since we need at least one mention per entity to trigger the rule
                    ent_matches = [(s_ent_id, sent_matches_by_id[i][0]) for i in sent_matches_by_ent[s_ent_id] if
                                   i in sent_matches_by_id]
                    if len(ent_matches) == 0:
                        cancel = True
                        break
                    all_ent_matches.extend(ent_matches)

                if cancel:
                    if debug_level >= 4:
                        print('   skip, did not find a mention for all entities')
                    continue

                # Sort from the largest to smallest matches
                all_ent_matches = sorted(all_ent_matches, key=lambda d: d[1]['end'] - d[1]['start'], reverse=True)

                if debug_level >= 4:
                    print('all_ent_matches', all_ent_matches)

                ###########
                ## WARNING
                ## This might behave in an unexpected manner when the mentions overlap
                ## The biggest mention will replace the smallest mention
                ## If two mentions overlap or one is contained by the other, then the smallest
                ## one will not appear at the end!
                ###########
                copy_matched_unigrams = deepcopy(matched_unigrams)
                # replace entity text by entity id
                for ent_i, (e_type, e) in enumerate(all_ent_matches):
                    if debug_level >= 4:
                        print('e : ', e)

                    # start_index and end_index are the indexes of the tokens: comes from spaCy after tokenization
                    # if start_index and end_index are known
                    # (the match comes from terms or entities using spacy results)
                    # use them to replace the matched tokens by the matched entity id
                    # This work only for the first entity to match, since after the first replacement,
                    # token indexes are changed
                    start_index = e['start_index']
                    if start_index is not None and ent_i == 0:
                        end_index = e['end_index'] + 1
                    # if they are not known
                    # (None, i.e. the matchs comes from regex applied to text and not to spacy)
                    # or if it is not the first mention that we try to replace
                    # then we need to find them
                    else:
                        # Parse tokens and find start_index and end_index based on character offsets
                        # (start and end)
                        char_start = e['start']
                        char_end = e['end']
                        token_indexes_to_replace = [i for i, term in enumerate(copy_matched_unigrams) if
                                                    term[0].start < char_end and term[0].end > char_start]
                        start_index = token_indexes_to_replace[0]
                        end_index = token_indexes_to_replace[-1] + 1

                    new_term = Term(e_type, copy_matched_unigrams[start_index][0].start,
                                    copy_matched_unigrams[end_index - 1][0].end,
                                    from_terminology=True, from_match=e['id'])

                    if debug_level >= 4:
                        print('add Term', new_term)
                    # print('start_index, ', start_index, end_index)
                    # print('matched_unigrams', matched_unigrams)
                    # print('1', Term(ent_id, copy_matched_unigrams[start_index][0].start, copy_matched_unigrams[end_index][0].end, from_terminology=True))
                    # print('copy_matched_unigrams[start]', [str(t) for t in copy_matched_unigrams[start_index]])
                    # print('copy_matched_unigrams[start][0]', copy_matched_unigrams[start_index][0])
                    copy_matched_unigrams[start_index:end_index] = [[new_term]]

                ##########
                ## Reiterate the entire process with a focus on the terms
                ## concerned by the rules
                ##########
                seed_term = '_'.join(sorted(candidate_second_level_expression))

                # no more than 6 recursions
                if recursive_call < 6:
                    sent_matches_by_id, sent_matches_by_ent, deleted_ids = \
                        self.match_in_sentence(sent_index,
                                               copy_matched_unigrams, sent_text, sent_start, sent_end,
                                               rule_dict, 
                                               doc_text,
                                               is_a_hierarchy,
                                               sections_df_row=sections_df_row,
                                               sent_matches_by_id=sent_matches_by_id,
                                               sent_matches_by_ent=sent_matches_by_ent,
                                               second_level_rules=second_level_rules, 
                                               best_match=best_match,
                                               seed_term=seed_term,
                                               recursive_call=recursive_call+1,
                                               debug_level=debug_level)
                    all_deleted_ids.update(deleted_ids)

            if debug_level >= 4:
                print(f'\n\nsent_matches after pass {recursive_call+2}\n\n', sent_matches_by_id)
        return sent_matches_by_id, sent_matches_by_ent, all_deleted_ids

    def match_regexes(self, text):
        '''
        @return dictionary representing the matches
        '''
        matches = []
        for regex, element_id in self.regexes.items():
            for match in regex.finditer(text):
                matches.append({"match_type": element_id, "start": match.start(), "end": match.end()})
        matches = pd.DataFrame(matches, columns=["match_type", "start", "end"]).sort_values(['start', 'end'])
        return matches

    def text_batch_to_df(self, text_paths, visit_ids, rule_dict, element_ids,
                         is_a_hierarchy=None, second_level_rules=None, custom_rule_functions=None,
                         brat_disabled_entities=None,
                         best_match=True, ignore_syntax=False, n_process=1, debug_level=0):
        if is_a_hierarchy is None:
            is_a_hierarchy = {}
        if second_level_rules is None:
            second_level_rules = {}
        if custom_rule_functions is None:
            custom_rule_functions = {}
        if brat_disabled_entities is None:
            brat_disabled_entities = set()
        logging.info('Collect {} texts'.format(len(visit_ids)))

        original_texts, texts = zip(*(path_2_text(text_path) for text_path in text_paths))
        
        #print('type gen', type(original_texts), type(texts))

        logging.info('Parse the documents with spaCy then apply the terminologies and rules')

        ehr_phenotyping_fields = ('ehr_phenotyping_id', 'visit_id', 'ehr_phenotyping_type', 'ehr_phenotyping_date',
                                  'ehr_phenotyping_datetime', 'ehr_phenotyping_value_as_string',
                                  'ehr_phenotyping_normalized_value_as_string', 
                                  'ehr_phenotyping_offset_start', 'ehr_phenotyping_offset_end',
                                  'ehr_phenotyping_value_as_number')
        ehr_relation_fields = ('ehr_phenotyping_relation_type',
                               'ehr_phenotyping_relation_date', 'ehr_phenotyping_relation_datetime',
                               'visit_id',
                               'ehr_phenotyping_source_id', 'ehr_phenotyping_target_id')
        ehr_phenotyping_ref_type_fields = ('ehr_phenotyping_type_id', 'ehr_phenotyping_type_short_name',
                                           'ehr_phenotyping_type_name')
        ehr_phenotyping_relation_type_fields = ('ehr_phenotyping_relation_id', 'ehr_phenotyping_relation_short_name',
                                                'ehr_phenotyping_relation_name')

        tab_all_ehr_phenotyping = []
        dict_all_ehr_phenotyping_types = {}
        tab_all_ehr_phenotyping_relations = []
        dict_all_ehr_phenotyping_relation_types = {}

        if n_process == 1:
            i = 0
            
            for parsed in tqdm(self.nlp.pipe(texts, n_process=n_process), total=len(visit_ids)):
                text = str(parsed)
                visit_id = visit_ids[i]
                original_text = original_texts[i]

                tab_ehr_phenotyping, tab_ehr_relations = self.text_to_df(text, visit_id, dict_all_ehr_phenotyping_types,
                                                                         dict_all_ehr_phenotyping_relation_types,
                                                                         rule_dict, element_ids, parsed=parsed, original_text=original_text,
                                                                         is_a_hierarchy=is_a_hierarchy,
                                                                         second_level_rules=second_level_rules,
                                                                         custom_rule_functions=custom_rule_functions,
                                                                         brat_disabled_entities=brat_disabled_entities,
                                                                         best_match=best_match,
                                                                         ignore_syntax=ignore_syntax,
                                                                         debug_level=debug_level)

                tab_all_ehr_phenotyping.extend(tab_ehr_phenotyping)
                tab_all_ehr_phenotyping_relations.extend(tab_ehr_relations)
                i += 1
        else:
            raise NotImplementedError()

        # Create df for elements
        df_ehr_phenotyping = pd.DataFrame(data=tab_all_ehr_phenotyping, columns=ehr_phenotyping_fields)
        tab_ehr_phenotyping_ref_type = [[type_id, type_name, type_name] for (type_name, type_id) in
                                        dict_all_ehr_phenotyping_types.items()]
        df_ehr_phenotyping_ref_type = pd.DataFrame(data=tab_ehr_phenotyping_ref_type,
                                                   columns=ehr_phenotyping_ref_type_fields)

        # create df for relations
        df_ehr_phenotyping_relation = pd.DataFrame(data=tab_all_ehr_phenotyping_relations,
                                                   columns=ehr_relation_fields)
        tab_ehr_phenotyping_relation_ref_type = [[type_id, type_name, type_name] for (type_name, type_id) in
                                                 dict_all_ehr_phenotyping_relation_types.items()]
        df_ehr_phenotyping_relation_ref_type = pd.DataFrame(data=tab_ehr_phenotyping_relation_ref_type,
                                                            columns=ehr_phenotyping_relation_type_fields)

        return df_ehr_phenotyping, df_ehr_phenotyping_ref_type, \
               df_ehr_phenotyping_relation, df_ehr_phenotyping_relation_ref_type

    def text_batch_to_brat(self, text_paths, ann_paths, rule_dict, element_ids,
                           is_a_hierarchy=None, second_level_rules=None, custom_rule_functions=None,
                           brat_disabled_entities=None,
                           best_match=True, ignore_syntax=False, n_process=1, debug_level=0):
        """
        """
        if is_a_hierarchy is None:
            is_a_hierarchy = {}
        if second_level_rules is None:
            second_level_rules = {}
        if custom_rule_functions is None:
            custom_rule_functions = []
        if brat_disabled_entities is None:
            brat_disabled_entities = set()
        # if n_process = 1, forget about apply_sync so that debugging is faster
        logging.info('Collect {} texts'.format(len(ann_paths)))

        original_texts, texts = zip(*(path_2_text(text_path) for text_path in text_paths))

        logging.info('Parse the documents with spaCy then apply the terminologies and rules')
        if n_process == 1:
            i = 0
            for parsed in tqdm(self.nlp.pipe(texts, n_process=n_process), total=len(ann_paths)):
                text = str(parsed)
                ann_path = ann_paths[i]
                original_text = original_texts[i]

                # print('++++++++++++++++++++')
                # print("ann_path", ann_path)

                self.text_to_brat(text, ann_path, rule_dict, element_ids, 
                                  parsed=parsed, original_text=original_text,
                                  is_a_hierarchy=is_a_hierarchy,
                                  second_level_rules=second_level_rules,
                                  custom_rule_functions=custom_rule_functions,
                                  brat_disabled_entities=brat_disabled_entities,
                                  best_match=best_match, ignore_syntax=ignore_syntax, debug_level=debug_level)
                i += 1
        else:
            raise NotImplementedError()  # ça marche pas...
            results = []
            with Pool(processes=n_process) as pool:
                pbar = tqdm(total=len(ann_paths))

                def update_pbar(*a):
                    pbar.update()

                i = 0
                for parsed in self.nlp.pipe(texts, n_process=n_process):
                    # print('Parse {}'.format(i))
                    text = str(parsed)
                    ann_path = ann_paths[i]
                    # res = pool.apipe(self.test, ('rrr', ))
                    print('res')
                    # results.append(res)

                    results.append(pool.apply_async(self.test, (33,),
                                                    callback=update_pbar))

                    # results.append(pool.apply_async(self.text_to_brat,
                    #                 (text, ann_path, rule_dict, element_ids),
                    #                 {'parsed':parsed, 'is_a_hierarchy':is_a_hierarchy, 'second_level_rules':second_level_rules,
                    #                  'brat_disabled_entities':brat_disabled_entities,
                    #                  'best_match':best_match, 'ignore_syntax':ignore_syntax, 'debug_level':debug_level},
                    #                 callback=update_pbar
                    #                ))
                    # self.text_to_brat(text, ann_path, rule_dict, element_ids, parsed=parsed, is_a_hierarchy=is_a_hierarchy, second_level_rules=second_level_rules,
                    #                  brat_disabled_entities=brat_disabled_entities,
                    #                  best_match=best_match, ignore_syntax=ignore_syntax, debug_level=debug_level)
                    i += 1
                    if i == 20:
                        break

                for res in results:
                    print(res.get())
                    # if not res.get():
                    #    raise NotImplementedError()
                print('close')
                pool.close()
                print('join')
                pool.join()
                print('done')

    def text_to_df(self, text, visit_id,
                   ehr_phenotyping_types, ehr_phenotyping_relation_types,
                   rule_dict, element_ids,
                   parsed=None, original_text=None, is_a_hierarchy=None,
                   second_level_rules=None, custom_rule_functions=None,
                   brat_disabled_entities=None,
                   best_match=True, ignore_syntax=False, debug_level=0):
        matches, sections = self.match(text, rule_dict,
                                       parsed=parsed, is_a_hierarchy=is_a_hierarchy,
                                       second_level_rules=second_level_rules,
                                       custom_rule_functions=custom_rule_functions,
                                       best_match=best_match,
                                       ignore_syntax=ignore_syntax, debug_level=debug_level)
        matches = self.transitive_closure(matches)

        # dataframes should contain the original text (if different)
        if original_text is not None:
            text = original_text
        
        ehr_phenotyping = []
        ehr_relations = []

        added_set = set()

        if brat_disabled_entities is None:
            brat_disabled_entities = set()
            
        # Add sections (if asked)
        if self.conf.KEEP_SECTIONS_IN_DATAFRAMES and sections is not None:
            for _, section in sections.iterrows():
                start = section['start']
                end = section['end']
                elem_id = self.conf.SECTION_PREFIX + section['match_type']                
                phen_type_id = ehr_phenotyping_types.get(elem_id, None)
                if phen_type_id is None:
                    phen_type_id = len(ehr_phenotyping_types)
                    ehr_phenotyping_types[elem_id] = phen_type_id
                k = '{}-{}-{}'.format(start, end, elem_id)
                new_phen = (
                k, visit_id, phen_type_id, datetime.date.today(), datetime.datetime.now(), text[start:end],
                elem_id, start, end, None)
                ehr_phenotyping.append(new_phen)
            
        # Build entities
        for k, match in matches.items():
            if self.conf.SECTION_PREFIX in k:
                continue
            for e in match:
                disabled = e['annotation_disabled']
                if not disabled:
                    start = e['start']
                    end = e['end']
                    elem_ids = e['type']
                    #elem_text = e['ngram']
                    # elem_id is already slugified
                    added_one = False
                    for elem_id in elem_ids:
                        if elem_id not in brat_disabled_entities:
                            phen_type_id = ehr_phenotyping_types.get(elem_id, None)
                            if phen_type_id is None:
                                phen_type_id = len(ehr_phenotyping_types)
                                ehr_phenotyping_types[elem_id] = phen_type_id

                            new_phen = (
                            k, visit_id, phen_type_id, datetime.date.today(), datetime.datetime.now(), text[start:end],
                            elem_id, start, end, None)
                            ehr_phenotyping.append(new_phen)
                            added_set.add(k)

        # Build relations
        for k in added_set:
            match = matches[k]
            for e in match:
                for rel_elem_id in e.get('related', []):
                    if rel_elem_id in matches and not matches[rel_elem_id][0][
                        'annotation_disabled'] and rel_elem_id in added_set:
                        phen_rel_type_id = ehr_phenotyping_relation_types.get('related', None)
                        if phen_rel_type_id is None:
                            phen_rel_type_id = len(ehr_phenotyping_relation_types)
                            ehr_phenotyping_relation_types['related'] = phen_rel_type_id

                        new_rel = (
                        phen_rel_type_id, datetime.date.today(), datetime.datetime.now(), visit_id, k, rel_elem_id)
                        ehr_relations.append(new_rel)

        return ehr_phenotyping, ehr_relations

    def text_to_brat(self, text, ann_path, rule_dict, element_ids,
                     parsed=None, original_text=None, is_a_hierarchy=None, second_level_rules=None, custom_rule_functions=None,
                     brat_disabled_entities=None, best_match=True, ignore_syntax=False, debug_level=0):
        matches, _ = self.match(text, rule_dict,
                                parsed=parsed, is_a_hierarchy=is_a_hierarchy,
                                second_level_rules=second_level_rules, custom_rule_functions=custom_rule_functions,
                                best_match=best_match, ignore_syntax=ignore_syntax, debug_level=debug_level)

        # Brat annotation should contain the original text (if different)
        if original_text is not None:
            text = original_text
        
        entity_index = 1
        attribute_index = 0
        # print('START')
        if brat_disabled_entities is None:
            brat_disabled_entities = set()
        with open(ann_path, 'w', encoding='utf-8') as f_out:
            # Write metadata element
            f_out.write('T1\t{} 0 0\t\n'.format(self.conf.METADATA_ENTITY_NAME))
            for _, match in matches.items():
                for e in match:
                    start = e['start']
                    end = e['end']
                    elem_ids = e['type']
                    #elem_text = e['ngram']
                    disabled = e['annotation_disabled']
                    if not disabled:
                        # elem_id is already slugified
                        for elem_id in elem_ids:
                            elem_type = element_ids[elem_id]['type']

                            if elem_id not in brat_disabled_entities:
                                entity_index += 1
                                if elem_type == schema.ENTITY:
                                    f_out.write(to_brat_entity(entity_index, elem_id, start, end, text[start:end]))
                                elif elem_type == schema.ATTRIBUTE:
                                    elem_parent, elem_key = element_ids[elem_id]['value']

                                    f_out.write(
                                        to_brat_entity(entity_index, slugify(elem_parent), start, end, text[start:end]))
                                    attribute_index += 1
                                    f_out.write(
                                        'A{}\t{} T{} {}\n'.format(attribute_index, slugify(elem_key), entity_index,
                                                                  slugify(elem_id)))
                                else:
                                    raise NotImplementedError()
        return True

    def match(self, text, rule_dict,
              parsed=None, is_a_hierarchy=None, second_level_rules=None, custom_rule_functions=None,
              best_match=True, ignore_syntax=False, debug_level=0):
        """Perform UMLS concept resolution for the given string.

        [extended_summary]

        Args:
            text (str): Text on which to run the algorithm

            best_match (bool, optional): Whether to return only the top match or all overlapping candidates. Defaults to True.
            ignore_syntax (bool, optional): Wether to use the heuristcs introduced in the paper (Soldaini and Goharian, 2016). TODO: clarify,. Defaults to False.

        Returns:
            List: List of all matches in the text
            TODO: Describe format
        """
        assert parsed is not None
        
        if self.term_searcher is None:
            self.term_searcher = Searcher(self.term_db, CosineMeasure())

        matches = dict()
        matches_by_ent = dict()
        sentence_starts = dict()
        all_deleted_ids = set()

        #########
        # Section splitting (document-level)
        #########
        if self.section_splitter is not None:
            sections_df = self.section_splitter.transform_text(text).sort_values(by=['start'])
        else:
            sections_df = None

        #########
        # Regex matching (document-level)
        #########
        match_regex_df = self.match_regexes(text)

        #########
        # Parsing + sentence-level matching
        #########
        if parsed is None:
            parsed = self.nlp('{}'.format(text))

        for sent_index, sent in enumerate(parsed.sents):
            # matched_unigrams is the list of all tokens of the sentence, where
            # matched unigrams have been replaced by their class and non-matched unigrams
            # have keen kept
            matched_unigrams = []
            # Find all unigrams matching with terminology search
            # if the unigram does not match, then keep the term as is
            for i in range(len(sent)):
                tok = sent[i:i + 1]
                if tok.text.strip() == "":
                    continue
                # Call search_token and get term result (index 0)
                token_matches = self.search_token(self.clean_token(tok.text))[0]
                terms = [item for sublist in
                         [[Term(m, tok.start_char, tok.end_char, from_terminology=True) for m in match[1]] for match in
                          token_matches] for item in sublist]
                terms_str = [item for sublist in
                             [[Term(m, tok.start_char, tok.end_char, from_terminology=True).__str__() for m in match[1]]
                              for match in token_matches] for item in sublist]
                if len(terms) == 0:
                    terms = [Term(tok.text, tok.start_char, tok.end_char,
                                  from_terminology=False, stopper=tok.text in self.conf.STAR_STOPPER)]
                # terms can have length > 1 if the token matched several terms from terminology
                matched_unigrams.append(terms)

            #####
            ## Split sentence if its spreads over several sections
            #####
            split_matched_unigrams = []
            if sections_df is None:
                split_matched_unigrams = [(None, matched_unigrams)]
            else:
                sent_sections_df = sections_df[
                    (sections_df['end'] >= sent.start_char) & (sections_df['start'] <= sent.end_char)]

                for _, row in sent_sections_df.iterrows():
                    start, end = row['start'], row['end']

                    # keep only terms that are in the new sentence
                    # if a word overlaps to the two parts (it happens!), 
                    # we consider that the word belongs to the second sentences
                    section_unigrams = [terms for terms in matched_unigrams if
                                        terms[0].start < end and terms[0].end > start and terms[0].end <= end]

                    if len(section_unigrams):
                        split_matched_unigrams.append((row, section_unigrams))

            all_sent_matches_by_id = dict()
            for (section_row, section_unigrams) in split_matched_unigrams:
                # Get sentence text
                if section_row is None:
                    sent_start_char = sent.start_char
                    sent_end_char = sent.end_char
                else:
                    sent_start_char = max(section_row['start'], sent.start_char)
                    sent_end_char = min(section_row['end'], sent.end_char)
                char_start = sent_start_char - sent.start_char
                char_end = sent_end_char - sent.start_char
                assert 0 <= char_start < char_end and char_end <= len(sent.text), \
                    '{}-{}-{}'.format(char_start, char_end, len(sent.text))
                sent_text = sent.text[char_start:char_end]

                #if "ganglions non métastatiques" not in sent_text:
                #    continue
                #debug_level = 4

                if debug_level >= 4:
                    print('-------------------')
                    print(sent_text)
                    if section_row is not None:
                        print('sent/section text', section_row['start'], sent.start_char, '-', section_row['end'], '-',
                              sent.text[char_start:char_end])
                    print('section_unigrams', section_unigrams)

                sent_matches_by_id, sent_matches_by_ent, deleted_ids = \
                    self.match_in_sentence(sent_index,
                                           section_unigrams, sent_text, sent_start_char, sent_end_char,
                                           rule_dict, text, is_a_hierarchy,
                                           sections_df_row=section_row,
                                           regexes_df=match_regex_df, sent_matches_by_id={}, sent_matches_by_ent={},
                                           second_level_rules=second_level_rules,
                                           best_match=best_match, debug_level=debug_level)
                assert not any([k in all_sent_matches_by_id for k in sent_matches_by_id]), 'Newly created ids are already in the match list: {}'.format([k for k in sent_matches_by_id if k in all_sent_matches_by_id])
                #print('all_deleted_ids', all_deleted_ids)

                all_sent_matches_by_id.update(sent_matches_by_id)
                for ent_type, ent_matches in sent_matches_by_ent.items():
                    match_set = matches_by_ent.get(ent_type, set())
                    match_set.update(ent_matches)
                    matches_by_ent[ent_type] = match_set

                sentence_starts[sent_index] = sent_start_char
                all_deleted_ids.update(deleted_ids)
                # convert dictionary id:list to list
            #########
            # Keep best (avoid overlap)
            if best_match:
                sent_matches = self._select_terms(all_sent_matches_by_id, is_a_hierarchy)
            else:
                sent_matches = all_sent_matches_by_id

            if debug_level >= 4:
                print('\nFINAL MATCHES')
                print(sent_matches)
                
            # Sanity check, the new matches should not 
            # already exist in the annotations
            # Two possible exceptions:
            # 1. mentions overlapping two sentences
            #    (coming from regexes) are tagged on both sides.
            #    They can be detected by the attributes 
            #    "extend_to_left_sentence" and "extend_to_right_sentence"
            # 2. mentions built by rules
            redundant_matches = [k for k in sent_matches if k in matches]
            for m_id in redundant_matches:
                #print('----------------------')
                #print('sent_matches', sent_matches[m_id])
                #print('matches', matches[m_id])
                if ('extend_to_left_sentence' in sent_matches[m_id][0] and \
                    'extend_to_right_sentence' in matches[m_id][0]):
                    continue
                elif ('origin' in sent_matches[m_id][0] and sent_matches[m_id][0]['origin'] == 'rule') and \
                   ('origin' in matches[m_id][0] and matches[m_id][0]['origin'] == 'rule'):
                    continue
                else:
                    raise Exception(f'Match {m_id} has been added but already exists!')
            #assert not any([k in matches for k in sent_matches])
            matches.update(sent_matches)

        ############
        # CUSTOM RULES (document-level)
        ############
        if custom_rule_functions is None:
            custom_rule_functions = []
        if debug_level >= 4 and len(custom_rule_functions) > 0:
            print('\nEND OF DOCUMENT, MATCHES before custom rules')
            print(matches)

        for rule_func in custom_rule_functions:
            # All custom rules must have the same parameters: matches_by_id and matches_by_ent_types
            # and return the same, updated variables
            matches, matches_by_ent = rule_func(matches, matches_by_ent)

        if debug_level >= 4 and len(custom_rule_functions) > 0:
            print('\nEND OF DOCUMENT, MATCHES after custom rules')
            print(matches)

        ############
        # Apply regular rules again
        #  (if we have custom rules)
        ############
        if len(custom_rule_functions):
            # reorganize matches by sentence in order to re-run the rules by sentence
            matches_by_sentences = {}
            for match_id, match in matches.items():
                sent_index = match[0]['sent_index']
                sent_matches = matches_by_sentences.get(sent_index, {})
                sent_matches[match_id] = match
                matches_by_sentences[sent_index] = sent_matches

            # apply rules to each sentence
            final_matches = dict()

            for sent_index, match_by_id_dict in matches_by_sentences.items():
                match_by_id_dict, matches_by_ent, deleted_ids = self.apply_rules(sent_index, match_by_id_dict, matches_by_ent,
                                                                    rule_dict,
                                                                    sent_text, sentence_starts[sent_index],
                                                                    is_a_hierarchy,
                                                                    add_intern_rules=True, debug_level=debug_level)
                final_matches.update(match_by_id_dict)
                all_deleted_ids.update(deleted_ids)
        else:
            final_matches = matches

        # Delete elements matching with a rule REMOVE (from apply_rules)
        for deleted_id in all_deleted_ids:
            if deleted_id in final_matches:
                del final_matches[deleted_id]
        if debug_level >= 4 and len(custom_rule_functions) > 0:
            print('\nEND OF DOCUMENT, FINAL MATCHES')
            print(final_matches)
        return final_matches, sections_df
