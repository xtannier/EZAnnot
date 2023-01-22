from tqdm import tqdm
import pandas as pd
import logging
import re

from . import configuration_sys as conf_sys
from .toolbox import slugify
from . import custom_boolean as boolean


def rule_error_message(file, index, message):
    """
    Returns an error message when parsing a raw leads to a format error
    """
    return '{} - {} - {}'.format(file, index + conf_sys.RULE_HEADER_HEIGHT + 1, message)


def protect_rule(rule_str, conf):
    """
    protect some spaces in the rule
    """
    result = ''
    in_symbol = ''
    rule_str = rule_str.replace('(', ' ( ').replace(')', ' ) ').replace('|', ' | ').replace('&', ' & ').replace('~',
                                                                                                                ' ~ ')
    rule_str = re.sub('\s+',' ',rule_str).replace('-', '_')
    for token in rule_str.split(' '):
        if token.lower() in conf_sys.ANY_ENTITY_MARKERS:
            result += conf_sys.ANY_ENTITY_TOKEN
        elif token.lower() in {'~', '&', '|', '(', ')',
                               conf.OP_AND.lower(), conf.OP_OR.lower(), conf.OP_NOT.lower(),
                               conf.OP_CONTAINS.lower(), conf.OP_IN.lower(), conf.OP_OVERLAPS.lower(),
                               '<', '>'}:
            result += in_symbol + ' ' + token
            in_symbol = ''
        else:
            if in_symbol == '':
                in_symbol = ' ' + token
            else:
                in_symbol += '_' + token
    result += ' ' + in_symbol
    return result


def index_rule(rule, rule_id, rule_action, ent_id, rule_dict, file_index, entity_ids, section_ids,
               in_excel_rules):
    """
    @param ent_id the slugified entity id
    """
    # check that all symbols are registered entity ids
    symbols = rule.get_symbols()
    unknown_symbols = [s for s in symbols if
                       slugify(str(s).strip()) not in (entity_ids.keys() | section_ids) and str(s).strip() != conf_sys.ANY_ENTITY_TOKEN]
    if len(unknown_symbols):
        logging.error(rule_error_message(in_excel_rules[file_index], file_index,
                                        'Symbols {} are not registered in the schema'.format(unknown_symbols)))
        return False

    literals = rule.get_literals()
    for lit in literals:
        if type(lit) == boolean.Symbol:
            s_lit = slugify(str(lit).strip())
            rules = rule_dict.get(s_lit, set())
            rules.add((rule_id, rule_action, ent_id, rule))
            rule_dict[s_lit] = rules
    return True

def check_rule(rule, match_dict, sent_matches_by_id, conf):
    """
    Check the validity of a given boolean rule.
    @param rule: the Boolean rule
    @param match_dict the term matches (by term)
    @param sent_matches_by_id the term (by id), which may have been
    modified by previous rules
    @return a (bool, list) pair. The boolean is True is the rule has been
    checked, False otherwise.
    The list contains the matchings that have been found
    during the rule parsing, whether
    """
    if type(rule) == boolean.Symbol:
        if str(rule) == conf_sys.ANY_ENTITY_TOKEN:
            # '*' matches all entities
            matching_ids = set()
            for e_id, e_m in match_dict.items():
                matching_ids.update(e_m)
        else:
            slug_rule = slugify(str(rule))
            matching_ids = match_dict.get(slug_rule, [])
        # select the matching ids that are still in the match list
        # (some may have been removed by previous rules)
        matching_ids = [matching_id for matching_id in matching_ids if matching_id in sent_matches_by_id]
        return len(matching_ids) > 0, matching_ids
    elif type(rule) == boolean.NOT:
        assert len(rule.args) == 1
        checked, match_list = check_rule(rule.args[0], match_dict, sent_matches_by_id, conf)
        return not checked, match_list
    elif type(rule) == boolean.AND:
        found_matches = []
        # Parse ALL arguments even if one is found to be false,
        # in order to gather all relevant matches into found_matches
        # so that we can relate them to each other later
        found = True
        for arg in rule.args:
            checked, match_list = check_rule(arg, match_dict, sent_matches_by_id, conf)
            if not checked:
                found = False
            found_matches.extend(match_list)
        return found, found_matches
    elif type(rule) == boolean.OR:
        found_matches = []
        # Parse ALL arguments even if one is found to be true,
        # in order to gather all relevant matches into found_matches
        # so that we can relate them to each other later
        found = False
        for arg in rule.args:
            checked, match_list = check_rule(arg, match_dict, sent_matches_by_id, conf)
            if checked:
                found = True
            found_matches.extend(match_list)
        return found, found_matches
    elif type(rule) == boolean.IN:
        arg1 = rule.args[0]
        checked1, match_list1 = check_rule(arg1, match_dict, sent_matches_by_id, conf)
        if not checked1:
            return False, match_list1
        # print('checked1')
        arg2 = rule.args[1]
        checked2, match_list2 = check_rule(arg2, match_dict, sent_matches_by_id, conf)
        if not checked2:
            return False, match_list2
        # print('checked2')
        match_list = []
        for match_id1 in match_list1:
            match1 = sent_matches_by_id[match_id1][0]
            start1, end1 = match1['start'], match1['end']
            for match_id2 in match_list2:
                if match_id1 != match_id2:
                    match2 = sent_matches_by_id[match_id2][0]
                    start2, end2 = match2['start'], match2['end']
                    # IN OK
                    if start1 >= start2 and end1 <= end2:
                        # The final match list contains only the left
                        # part of the expression ("A in B" returns A)
                        match_list.append(match_id1)
                        break
        return len(match_list) > 0, match_list
    elif type(rule) == boolean.CONTAINS:
        arg1 = rule.args[0]
        checked1, match_list1 = check_rule(arg1, match_dict, sent_matches_by_id, conf)
        # print('-----------------------------------------')
        # print('CONTAINS', rule)
        # print('sent_matches_by_id', sent_matches_by_id)
        if not checked1:
            return False, match_list1
        # print('checked1')
        arg2 = rule.args[1]
        checked2, match_list2 = check_rule(arg2, match_dict, sent_matches_by_id, conf)
        if not checked2:
            return False, match_list2
        # print('checked2')
        match_list = []
        for match_id1 in match_list1:
            match1 = sent_matches_by_id[match_id1][0]
            start1, end1 = match1['start'], match1['end']
            for match_id2 in match_list2:
                if match_id1 != match_id2:
                    match2 = sent_matches_by_id[match_id2][0]
                    start2, end2 = match2['start'], match2['end']
                    # contains OK
                    if start1 <= start2 and end1 >= end2:
                        # The final match list contains only the left
                        # part of the expression ("A CONTAINS B" returns A)
                        match_list.append(match_id1)
                        break
        return len(match_list) > 0, match_list
    elif type(rule) == boolean.OVERLAPS:
        arg1 = rule.args[0]
        checked1, match_list1 = check_rule(arg1, match_dict, sent_matches_by_id, conf)
        if not checked1:
            return False, match_list1
        # print('checked1')
        arg2 = rule.args[1]
        checked2, match_list2 = check_rule(arg2, match_dict, sent_matches_by_id, conf)
        if not checked2:
            return False, match_list2
        match_list = []
        for match_id1 in match_list1:
            match1 = sent_matches_by_id[match_id1][0]
            start1, end1 = match1['start'], match1['end']
            for match_id2 in match_list2:
                match2 = sent_matches_by_id[match_id2][0]
                if match1['type'] != match2['type']:
                    start2, end2 = match2['start'], match2['end']
                    # overlaps OK
                    if not (end1 < start2 or end2 < start1):
                        # The final match list contains only the left
                        # part of the expression ("A OVERLAPS B" returns A)
                        match_list.append(match_id1)
        return len(match_list) > 0, match_list
    else:
        raise ValueError(rule)


def load_rules(in_excel_rules, rule_parser, ent_ids, section_ids, conf, disable_tqdm=False):
    """
    Loads the rules from Excel files
    """
    rule_dfs = []

    for excel_file in tqdm(in_excel_rules):
        # Read Excel file
        df = pd.read_excel(excel_file, header=None,
                           keep_default_na=False,
                           names=conf_sys.RULE_COLUMN_NAME,
                           usecols=[i for i in range(len(conf_sys.RULE_COLUMN_NAME))],
                           skiprows=[i for i in range(conf_sys.RULE_HEADER_HEIGHT)])
        # Do NOT remove empty lines in order to keep line numbers as in the original file
        rule_dfs.append(df)

    # Rule_dict contains an index to find rules concerned by a (not negated) symbol.
    rule_dict = {}

    rule_id = 0
    total_lines = sum([len(df) for df in rule_dfs])
    pbar = tqdm(total=total_lines, disable=disable_tqdm)

    error = 0
    
    for file_index, rule_df in enumerate(rule_dfs):
        for index, row in rule_df.iterrows():
            ent_id = row[conf_sys.RULE_ID].strip()
            s_ent_id = slugify(ent_id)
            rule_str = row[conf_sys.RULE].strip()
            rule_action = row[conf_sys.ACTION].strip()
            pbar.update()

            if len(rule_str.strip()) == 0:
                continue
            if rule_action not in conf.RULE_ACTIONS:
                logging.error(rule_error_message(in_excel_rules[file_index], index,
                                                     'Action must be in {}'.format(conf.RULE_ACTIONS)))
                error += 1
                continue
            if rule_action != conf.RULE_ADD and s_ent_id not in ent_ids and s_ent_id not in section_ids and s_ent_id != conf_sys.ANY_ENTITY_TOKEN:
                logging.error(rule_error_message(in_excel_rules[file_index], index,
                                                    'Entity {} is not registered in the schema'.format(ent_id)))
                error += 1
                continue

            if rule_action == conf.RULE_ADD:
                if ent_id.startswith(conf.SECTION_PREFIX):
                    section_ids.add(ent_id)
            try:
                # REMOVE and LINK rules are a combination of left column and right column of the Excel file
                if rule_action != conf.RULE_ADD:
                    # IN/CONTAINS/OVERLAPS
                    # -> build a binary relation between left and right
                    if rule_str.upper().startswith(conf.OP_IN.upper()) or \
                            rule_str.upper().startswith(conf.OP_CONTAINS.upper()) or \
                            rule_str.upper().startswith(conf.OP_OVERLAPS.upper()):

                        rule_str = ent_id + ' ' + rule_str
                        rule = rule_parser.parse(protect_rule(rule_str, conf))
                    # Other or not operator
                    # -> left AND right
                    else:
                        rule_left = rule_parser.parse(protect_rule(ent_id, conf))
                        rule_right = rule_parser.parse(protect_rule(rule_str, conf))
                        rule = boolean.AND(rule_left, rule_right)
                # ADD rules are just the right column... and if it is checked, create
                # the element of the left column
                else:
                    rule = rule_parser.parse(protect_rule(rule_str, conf))
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise SyntaxError(rule_error_message(in_excel_rules[file_index], index,
                                                     'Invalid expression {}'.format(rule_str)))

            res = index_rule(rule, rule_id, rule_action, s_ent_id, rule_dict, file_index, ent_ids, section_ids,
                       in_excel_rules)
            if not res:
                error += 1
                continue
            rule_id += 1
    pbar.close()
    
    if error > 0:
        raise ValueError(f'{error} line(s) in the rules contained an error, '
                         'see error messages above')
    
    return rule_dict
