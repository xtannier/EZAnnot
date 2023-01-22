from boolean.boolean import BooleanAlgebra, DualBase, Symbol, ParseError, \
    _TRUE, _FALSE, NOT, AND, OR, TOKEN_TYPES, TOKEN_TRUE, TOKEN_FALSE, \
    TOKEN_LPAR, TOKEN_RPAR, TOKEN_OR, TOKEN_AND, TOKEN_NOT, TRACE_PARSE, TOKEN_SYMBOL, \
    PARSE_INVALID_OPERATOR_SEQUENCE, PARSE_INVALID_SYMBOL_SEQUENCE, PARSE_INVALID_EXPRESSION, PARSE_UNKNOWN_TOKEN, \
    PARSE_INVALID_NESTING, PARSE_UNBALANCED_CLOSING_PARENS, \
    inspect, Function

TOKEN_IN = 9
TOKEN_CONTAINS = 10
TOKEN_OVERLAPS = 11

TOKEN_TYPES.update({TOKEN_IN: 'IN', TOKEN_CONTAINS: 'CONTAINS', TOKEN_OVERLAPS: 'OVERLAPS'})

# Python 2 and 3
try:
    basestring  # NOQA
except NameError:
    basestring = str  # NOQA


class OVERLAPS(DualBase):
    """
    Boolean IN operation, taking 2 or more arguments.
    It can also be created by using "IN" between two boolean expressions.
    """

    # sort_order = 10
    # _pyoperator = and_operator

    def __init__(self, arg1, arg2, *args):
        super(OVERLAPS, self).__init__(arg1, arg2, *args)
        self.identity = None
        self.annihilator = None
        self.dual = None
        self.operator = '$'


class CONTAINS(DualBase):
    """
    Boolean CONTAINS operation, taking 2 or more arguments.
    It can also be created by using "IN" between two boolean expressions.
    """

    # sort_order = 10
    # _pyoperator = and_operator

    def __init__(self, arg1, arg2, *args):
        super(CONTAINS, self).__init__(arg1, arg2, *args)
        self.identity = None
        self.annihilator = None
        self.dual = self.IN
        self.operator = '>'


class IN(DualBase):
    """
    Boolean IN operation, taking 2 or more arguments.
    It can also be created by using "IN" between two boolean expressions.
    """

    # sort_order = 10
    # _pyoperator = and_operator

    def __init__(self, arg1, arg2, *args):
        super(IN, self).__init__(arg1, arg2, *args)
        self.identity = None
        self.annihilator = None
        self.dual = self.CONTAINS
        self.operator = '<'


class CustomBooleanAlgebra(BooleanAlgebra):
    def __init__(self, TRUE_class=None, FALSE_class=None, Symbol_class=None,
                 NOT_class=None, AND_class=None, OR_class=None, IN_class=None, CONTAINS_class=None,
                 OVERLAPS_class=None,
                 allowed_in_token=('.', ':', '_', '*')):
        """
        The types for TRUE, FALSE, NOT, AND, OR and Symbol define the boolean
        algebra elements, operations and Symbol variable. They default to the
        standard classes if not provided.
        You can customize an algebra by providing alternative subclasses of the
        standard types.
        """
        # TRUE and FALSE base elements are algebra-level "singleton" instances
        self.TRUE = TRUE_class or _TRUE
        self.TRUE = self.TRUE()

        self.FALSE = FALSE_class or _FALSE
        self.FALSE = self.FALSE()

        # they cross-reference each other
        self.TRUE.dual = self.FALSE
        self.FALSE.dual = self.TRUE

        # boolean operation types, defaulting to the standard types
        self.NOT = NOT_class or NOT
        self.AND = AND_class or AND
        self.OR = OR_class or OR

        self.IN = IN_class or IN
        self.CONTAINS = CONTAINS_class or CONTAINS
        self.OVERLAPS = OVERLAPS_class or OVERLAPS

        # class used for Symbols
        self.Symbol = Symbol_class or Symbol

        tf_nao = {
            'TRUE': self.TRUE,
            'FALSE': self.FALSE,
            'NOT': self.NOT,
            'AND': self.AND,
            'OR': self.OR,
            'IN': self.IN,
            'CONTAINS': self.CONTAINS,
            'OVERLAPS': self.OVERLAPS,
            'Symbol': self.Symbol
        }

        # setup cross references such that all algebra types and
        # objects hold a named attribute for every other types and
        # objects, including themselves.
        for obj in tf_nao.values():
            for name, value in tf_nao.items():
                setattr(obj, name, value)

        # Set the set of characters allowed in tokens
        self.allowed_in_token = allowed_in_token

    def definition(self):
        """
        Return a tuple of this algebra defined elements and types as:
        (TRUE, FALSE, NOT, AND, OR, Symbol)
        """
        return self.TRUE, self.FALSE, self.NOT, self.AND, self.OR, self.IN, self.CONTAINS, self.OVERLAPS, self.Symbol

    def tokenize(self, expr):
        """
        Return an iterable of 3-tuple describing each token given an expression
        unicode string.
        This 3-tuple contains (token, token string, position):
        - token: either a Symbol instance or one of TOKEN_* token types.
        - token string: the original token unicode string.
        - position: some simple object describing the starting position of the
          original token string in the `expr` string. It can be an int for a
          character offset, or a tuple of starting (row/line, column).
        The token position is used only for error reporting and can be None or
        empty.
        Raise ParseError on errors. The ParseError.args is a tuple of:
        (token_string, position, error message)
        You can use this tokenizer as a base to create specialized tokenizers
        for your custom algebra by subclassing BooleanAlgebra. See also the
        tests for other examples of alternative tokenizers.
        This tokenizer has these characteristics:
        - The `expr` string can span multiple lines,
        - Whitespace is not significant.
        - The returned position is the starting character offset of a token.
        - A TOKEN_SYMBOL is returned for valid identifiers which is a string
        without spaces. These are valid identifiers:
            - Python identifiers.
            - a string even if starting with digits
            - digits (except for 0 and 1).
            - dotted names : foo.bar consist of one token.
            - names with colons: foo:bar consist of one token.
            These are not identifiers:
            - quoted strings.
            - any punctuation which is not an operation
        - Recognized operators are (in any upper/lower case combinations):
            - for and:  '*', '&', 'and'
            - for or: '+', '|', 'or'
            - for not: '~', '!', 'not'
        - Recognized special symbols are (in any upper/lower case combinations):
            - True symbols: 1 and True
            - False symbols: 0, False and None
        """
        if not isinstance(expr, basestring):
            raise TypeError('expr must be string but it is %s.' % type(expr))

        # mapping of lowercase token strings to a token type id for the standard
        # operators, parens and common true or false symbols, as used in the
        # default tokenizer implementation.
        TOKENS = {
            '*': TOKEN_AND, '&': TOKEN_AND, 'and': TOKEN_AND,
            '+': TOKEN_OR, '|': TOKEN_OR, 'or': TOKEN_OR,
            '~': TOKEN_NOT, '!': TOKEN_NOT, 'not': TOKEN_NOT,
            'in': TOKEN_IN, '<': TOKEN_IN,
            'contains': TOKEN_CONTAINS, '>': TOKEN_CONTAINS,
            'overlaps': TOKEN_OVERLAPS, '$': TOKEN_OVERLAPS,
            '(': TOKEN_LPAR, ')': TOKEN_RPAR,
            '[': TOKEN_LPAR, ']': TOKEN_RPAR,
            'true': TOKEN_TRUE, '1': TOKEN_TRUE,
            'false': TOKEN_FALSE, '0': TOKEN_FALSE, 'none': TOKEN_FALSE
        }

        position = 0
        length = len(expr)

        while position < length:
            tok = expr[position]

            sym = tok.isalnum() or tok == '_'
            if sym:
                position += 1
                while position < length:
                    char = expr[position]
                    if char.isalnum() or char in self.allowed_in_token:
                        position += 1
                        tok += char
                    else:
                        break
                position -= 1

            try:
                yield TOKENS[tok.lower()], tok, position
            except KeyError:
                if sym:
                    yield TOKEN_SYMBOL, tok, position
                elif tok not in (' ', '\t', '\r', '\n'):
                    raise ParseError(token_string=tok, position=position,
                                     error_code=PARSE_UNKNOWN_TOKEN)

            position += 1

    def parse(self, expr, simplify=False):
        """
        Return a boolean expression parsed from `expr` either a unicode string
        or tokens iterable.
        Optionally simplify the expression if `simplify` is True.
        Raise ParseError on errors.
        If `expr` is a string, the standard `tokenizer` is used for tokenization
        and the algebra configured Symbol type is used to create Symbol
        instances from Symbol tokens.
        If `expr` is an iterable, it should contain 3-tuples of: (token_type,
        token_string, token_position). In this case, the `token_type` can be
        a Symbol instance or one of the TOKEN_* constant types.
        See the `tokenize()` method for detailed specification.
        """

        precedence = {self.NOT: 5, self.IN: 7, self.CONTAINS: 7, self.OVERLAPS: 7, self.AND: 10, self.OR: 15,
                      TOKEN_LPAR: 20}

        if isinstance(expr, basestring):
            tokenized = self.tokenize(expr)
        else:
            tokenized = iter(expr)

        if TRACE_PARSE:
            tokenized = list(tokenized)
            print('tokens:')
            for t in tokenized:
                print(t)
            tokenized = iter(tokenized)

        # the abstract syntax tree for this expression that will be build as we
        # process tokens
        # the first two items are None
        # symbol items are appended to this structure
        ast = [None, None]

        def is_sym(_t):
            return isinstance(_t, Symbol) or _t in (TOKEN_TRUE, TOKEN_FALSE, TOKEN_SYMBOL)

        def is_operator(_t):
            return _t in (TOKEN_AND, TOKEN_OR, TOKEN_IN, TOKEN_CONTAINS, TOKEN_OVERLAPS)

        prev_token = None
        for token_type, token_string, token_position in tokenized:
            if TRACE_PARSE:
                print('\nprocessing token_type:', repr(token_type), 'token_string:', repr(token_string),
                      'token_position:', repr(token_position))

            if prev_token:
                prev_token_type, _prev_token_string, _prev_token_position = prev_token
                if TRACE_PARSE:
                    print('  prev_token:', repr(prev_token))

                if is_sym(prev_token_type) and (is_sym(token_type)):  # or token_type == TOKEN_LPAR) :
                    raise ParseError(token_type, token_string, token_position, PARSE_INVALID_SYMBOL_SEQUENCE)

                if is_operator(prev_token_type) and (is_operator(token_type) or token_type == TOKEN_RPAR):
                    raise ParseError(token_type, token_string, token_position, PARSE_INVALID_OPERATOR_SEQUENCE)

            else:
                if is_operator(token_type):
                    raise ParseError(token_type, token_string, token_position, PARSE_INVALID_OPERATOR_SEQUENCE)

            if token_type == TOKEN_SYMBOL:
                ast.append(self.Symbol(token_string))
                if TRACE_PARSE:
                    print(' ast: token_type is TOKEN_SYMBOL: append new symbol', repr(ast))

            elif isinstance(token_type, Symbol):
                ast.append(token_type)
                if TRACE_PARSE:
                    print(' ast: token_type is Symbol): append existing symbol', repr(ast))

            elif token_type == TOKEN_TRUE:
                ast.append(self.TRUE)
                if TRACE_PARSE: print(' ast: token_type is TOKEN_TRUE:', repr(ast))

            elif token_type == TOKEN_FALSE:
                ast.append(self.FALSE)
                if TRACE_PARSE: print(' ast: token_type is TOKEN_FALSE:', repr(ast))

            elif token_type == TOKEN_NOT:
                ast = [ast, self.NOT]
                if TRACE_PARSE: print(' ast: token_type is TOKEN_NOT:', repr(ast))

            elif token_type == TOKEN_IN:
                ast = self._start_operation(ast, self.IN, precedence)
                if TRACE_PARSE:
                    print('  ast:token_type is TOKEN_IN: start_operation', ast)

            elif token_type == TOKEN_CONTAINS:
                ast = self._start_operation(ast, self.CONTAINS, precedence)
                if TRACE_PARSE:
                    print('  ast:token_type is TOKEN_CONTAINS: start_operation', ast)

            elif token_type == TOKEN_OVERLAPS:
                ast = self._start_operation(ast, self.OVERLAPS, precedence)
                if TRACE_PARSE:
                    print('  ast:token_type is TOKEN_OVERLAPS: start_operation', ast)

            elif token_type == TOKEN_AND:
                ast = self._start_operation(ast, self.AND, precedence)
                if TRACE_PARSE:
                    print('  ast:token_type is TOKEN_AND: start_operation', ast)

            elif token_type == TOKEN_OR:
                ast = self._start_operation(ast, self.OR, precedence)
                if TRACE_PARSE:
                    print('  ast:token_type is TOKEN_OR: start_operation', ast)

            elif token_type == TOKEN_LPAR:
                if prev_token:
                    # Check that an opening parens is preceded by a function
                    # or an opening parens
                    if prev_token_type not in (
                    TOKEN_NOT, TOKEN_AND, TOKEN_OR, TOKEN_IN, TOKEN_CONTAINS, TOKEN_OVERLAPS, TOKEN_LPAR):
                        raise ParseError(token_type, token_string, token_position, PARSE_INVALID_NESTING)
                ast = [ast, TOKEN_LPAR]

            elif token_type == TOKEN_RPAR:
                while True:
                    if ast[0] is None:
                        raise ParseError(token_type, token_string, token_position, PARSE_UNBALANCED_CLOSING_PARENS)

                    if ast[1] is TOKEN_LPAR:
                        ast[0].append(ast[2])
                        if TRACE_PARSE: print('ast9:', repr(ast))
                        ast = ast[0]
                        if TRACE_PARSE: print('ast10:', repr(ast))
                        break

                    if isinstance(ast[1], int):
                        raise ParseError(token_type, token_string, token_position, PARSE_UNBALANCED_CLOSING_PARENS)

                    # the parens are properly nested
                    # the top ast node should be a function subclass
                    if not (inspect.isclass(ast[1]) and issubclass(ast[1], Function)):
                        raise ParseError(token_type, token_string, token_position, PARSE_INVALID_NESTING)

                    subex = ast[1](*ast[2:])
                    ast[0].append(subex)
                    if TRACE_PARSE: print('ast11:', repr(ast))
                    ast = ast[0]
                    if TRACE_PARSE: print('ast12:', repr(ast))
            else:
                raise ParseError(token_type, token_string, token_position, PARSE_UNKNOWN_TOKEN)

            prev_token = (token_type, token_string, token_position)

        try:
            while True:
                if ast[0] is None:
                    if TRACE_PARSE: print('ast[0] is None:', repr(ast))
                    if ast[1] is None:
                        if TRACE_PARSE: print('  ast[1] is None:', repr(ast))
                        if len(ast) != 3:
                            raise ParseError(error_code=PARSE_INVALID_EXPRESSION)
                        parsed = ast[2]
                        if TRACE_PARSE: print('    parsed = ast[2]:', repr(parsed))

                    else:
                        # call the function in ast[1] with the rest of the ast as args
                        parsed = ast[1](*ast[2:])
                        if TRACE_PARSE: print('  parsed = ast[1](*ast[2:]):', repr(parsed))
                    break
                else:
                    if TRACE_PARSE: print('subex = ast[1](*ast[2:]):', repr(ast))
                    subex = ast[1](*ast[2:])
                    ast[0].append(subex)
                    if TRACE_PARSE: print('  ast[0].append(subex):', repr(ast))
                    ast = ast[0]
                    if TRACE_PARSE: print('    ast = ast[0]:', repr(ast))
        except TypeError:
            raise ParseError(error_code=PARSE_INVALID_EXPRESSION)

        if simplify:
            return parsed.simplify()

        if TRACE_PARSE: print('final parsed:', repr(parsed))
        return parsed
