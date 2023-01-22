import re

class Configuration(object):
    def __init__(self):

        # Debug level
        self.debug_level = 3

        ###########
        # Spacy model
        # self.SPACY_MODEL = "fr_core_news_sm"
        self.SPACY_MODEL = '/export/home/opt/data/spacy/fr_core_news_sm-2.1.0/fr_core_news_sm/fr_core_news_sm-2.1.0'
        ###########

        ###########
        # Schema
        ###########
        self.ATTRIBUTE_SEPARATOR = '|'
        self.KEY_VALUE_SEPARATOR = ":"
        self.PATH_SEPARATOR = '>'
        # Allow short names in Brat configuration (short names are automatically infered so may be confusing)
        self.ALLOW_SHORT_NAMES = False

        ###########
        # Search
        ###########
        self.SIMILARITY_THRESHOLD = 0.80

        ###########
        # Section splitting
        ###########
        # Type of the section splitter (currently: None for no section splitting, section_v1, section_v2)
        self.SECTION_SPLITTER = 'section_v2'

        # Prefix used to start section names in the terminology & rule files and in the annotations
        self.SECTION_PREFIX = 'SECTION_'
        # Types of section provided by the section splitter
        self.SECTION_TYPES = {'head', 'motif', 'antecedent', 'antecedent_familiaux', 'mode_de_vie', 'traitement_entree',
                              'histoire', 'examen_clinique', 'examen_complementaire', 'evolution', 'conclusion',
                              'traitement_sortie', 'traitement', 'autre'}
        
        # Do you want to keep the section information in the dataframe output?
        # (you'll have a lot of sections, hence a lot of extra raws in the df)
        self.KEEP_SECTIONS_IN_DATAFRAMES = False

        ###########
        # Terminology and rules
        # Do not edit this part unless you need the reserved tokens in your terminologies
        # (e.g. if you want to add CONTAINS in your terminology, 
        #  you have to choose another keyword for the CONTAINS relations)
        ###########
        # Jokers
        self.WORD_JOKER_UP_TO_5 = ['___']                       # replace between 0 and 5 tokens
        self.WORD_JOKER_UP_TO_3 = ['...', '....', '…', '….']    # replace between 0 and 3 tokens
        self.WORD_JOKER_UP_TO_N = ['*', '>']                    # replace between 0 and n tokens
        self.WORD_JOKER = ['.']                                 # replace 1 token
        self.WORD_OPTIONAL_JOKER = ['?']                        # replace 0 or 1 token
        self.NUM_JOKER = '<NUM>'                                # replace a number
        self.MAX_N = 15                            # * joker cannot extend over more than MAX_N tokens (-1 for no limit)
        # Stopper for jokers: these tokens will end the term expansion, even if there is a joker
        self.STAR_STOPPER = {'.', ',', ';', '(', ')', ':', '!', '?'}
        # This operator in the expression means "left and right operands must appear in the same sentence"
        self.SAME_SENTENCE_OPERATOR = '<>'

        # Tokens used for operators in rules
        self.OP_NOT = 'NOT'
        self.OP_AND = 'AND'
        self.OP_OR = 'OR'
        self.OP_IN = 'IN'
        self.OP_CONTAINS = 'CONTAINS'
        self.OP_OVERLAPS = 'OVERLAPS'

        # Type of rule actions
        self.RULE_ACTIONS = [self.RULE_REMOVE, self.RULE_ADD, self.RULE_LINK] = ['REMOVE', 'ADD', 'LINK']

        # Regex to match NUM_JOKER
        self.NUM_REGEX = '\d+([,.]\d+)?'

        # These characters must ALWAYS be token separators
        self.EXTRA_TOKEN_SEPARATORS = [':', '=', '/']           # To add to spaCy's infix_finditer

        ##########
        # Brat
        ##########
        self.METADATA_ENTITY_NAME = 'metadata'
        self.METADATA_ANNOTATION_DONE_ATTRIBUTE = 'done'

        # Do not edit
        self.BRAT_DONE_REGEX = re.compile('A\d+\t{} T(\d+)'.format(self.METADATA_ANNOTATION_DONE_ATTRIBUTE))
        self.BRAT_METADATA_REGEX = re.compile('T(\d+)\t{} \d+ \d+\t.*'.format(self.METADATA_ENTITY_NAME))


