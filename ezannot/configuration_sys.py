# Types that can be given to expressions in the Excel terminology files
# ENTITY: exact match needed
# TERM: approximate match allowed
# REGEX: regular expression
# EXCLUSION: exclusion list for term matching the terminology but that should be excluded anyway
EXPRESSION_TYPES = [ENTITY, TERM, REGEX, EXCLUSION] = ['entity', 'term', 'regex', 'exclusion']

# Above this word length, approximate search is allowed
APPROX_MATCHING_LENGTH_MIN = 4

######
# RULES
######
# Names of the columns in the rule Excel files
RULE_COLUMN_NAME = [RULE_ID, RULE, ACTION] = ['id', 'rule', 'action']
# Number of raws occupied by the headers in the rule Excel files
RULE_HEADER_HEIGHT = 1

######
# TERMINOLOGIES
######
# Names of the columns in the terminology Excel files
TERMINO_COLUMN_NAMES = [TERMINO_ID, EXPRESSION, EXPR_TYPE, PREANNOTATION] = \
    ['id', 'expression', 'expr_type', 'preannotation']
# Number of raws occupied by the headers in the terminology Excel files
TERMINO_HEADER_HEIGHT = 1
# Terms allowed for enabling or disabling entity annotation
DISABLED_TERMS = {'': False, 'oui': False, 'yes': False, 'non': True, 'no': True,
                  1: False, 0: True, '1': False, '0': True}

# These character strings are used to represent "any entity" in the rule definition Excel file
ANY_ENTITY_MARKERS = {'ANY', '*'}
# Token used internally to represent "any entity" in the rules
# (must be changed if needs to be used for a real entity name)
ANY_ENTITY_TOKEN = '__ANY__'
