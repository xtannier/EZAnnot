import logging
import pandas as pd
from .toolbox import slugify

COLUMN_NAMES = [ID, PATH, DESC, BRAT_TYPE, BRAT_LEVEL, BRAT_ENABLED] = ['id', 'path', 'description', 'brat_type',
                                                                        'brat_level', 'brat_enabled']
DISABLED_CHOICES = {'': False, 'oui': False, 'yes': False, 'non': True, 'no': True, 1: False, 0: True, '1': False,
                    '0': True}
BRAT_TYPES = [ENTITY, ATTRIBUTE, EVENT] = ['entity', 'attribute', 'event']
DEFAULT_ATTRIBUTE_SUFFIX = 'type'
HEADER_HEIGHT = 2


def error_message(file, index, message):
    return '{} - {} - {}'.format(file, index + HEADER_HEIGHT + 1, message)


def get_parent_path(path, file, index, conf, brat_level=-1):
    '''
    Splits the path in two : (hierarchy parent path, Brat parent path, base name)
    The hierarchy parent path is the "real" parent path (whole path to the entity) -> list of ancestors
    The Brat parent path depends on the hierarchy index provided in the Excel file (e.g. None if hierarchy is 1) -> joined list
    '''
    path_split = path.split(conf.PATH_SEPARATOR)
    # No parent
    if len(path_split) == 1:
        if brat_level > 1:
            raise ValueError(
                error_message(file, index, 'Brat level {} is not compatible with path {}'.format(brat_level, path)))
        return [], None, path
    else:
        # At least 1 parent
        if brat_level == -1:
            brat_level = len(path_split)
        if brat_level == 1:
            return path_split[:-1], None, path_split[-1]
        elif brat_level > len(path_split):
            raise ValueError(
                error_message(file, index, 'Brat level {} is not compatible with path {}'.format(brat_level, path)))
        else:
            # print('  return ', PATH_SEPARATOR.join(path_split[:brat_level-1]), path_split[-1])
            return path_split[:-1], conf.PATH_SEPARATOR.join(path_split[:brat_level - 1]), path_split[-1]


def get_brat_level(row):
    level = row[BRAT_LEVEL]
    if level == '':
        return -1
    else:
        return int(level)


def add_attribute(file, index, att_id, entity_path, key, value, attributes, all_ids, all_entity_paths, path_to_id, conf):
    """
    Add the attribute att_id to entity entity_path.
    @param att_id the (slugified) attribute id
    """
    # if the entity path is an entity, then the attribute is an attribute of this entity
    if entity_path in all_entity_paths:
        _, _, entity_name = get_parent_path(entity_path, file, index, conf)
        path_id = path_to_id[entity_path]
        s_entity_name = slugify(entity_name)
        if key is None:
            key = s_entity_name + '_' + DEFAULT_ATTRIBUTE_SUFFIX
        attribute_dict = attributes.get(key, {})
        values = attribute_dict.get(path_id, [])
        if value is None and not (len(values) == 0 or values == [None]):
            raise SyntaxError(error_message(file,
                                            index,
                                            'attribute {} already has values, '
                                            'cannot be a binary attribute'.format(att_id)))
        elif att_id in values:
            raise SyntaxError(error_message(file, index, 'duplicate attribute {}:{}'.format(key, att_id)))
        if value is None and len(values) == 0:
            values.append(value)
        else:
            values.append(att_id)
        attribute_dict[path_id] = values
        attributes[key] = attribute_dict
        assert value not in all_ids
        all_ids[att_id] = {"type": ATTRIBUTE, "value": (path_id, key)}
    # Otherwise we go back up to the parent
    else:
        _, brat_parent_path, _ = get_parent_path(entity_path, file, index, conf)
        if brat_parent_path is None:
            raise SyntaxError(error_message(file,
                                            index,
                                            'Could not find entity corresponding '
                                            'to the attribute {}:{}'.format(key, value)))

        add_attribute(file, index, att_id, brat_parent_path, key, value, attributes, all_ids, all_entity_paths, path_to_id, conf)


def build_section_df(conf):
    section_tab = []
    # 1. high-level SECTION entity
    section_tab.append([conf.SECTION_PREFIX, conf.SECTION_PREFIX, '', 'entity', 1, False])
    # 2. second-level SECTION entities
    for s in conf.SECTION_TYPES:
        s = conf.SECTION_PREFIX + s
        section_tab.append([s, conf.SECTION_PREFIX + conf.PATH_SEPARATOR + s, '', 'entity', 2, True])
    section_df = pd.DataFrame(data=section_tab, columns=COLUMN_NAMES)
    return section_df


def parse_schema_excel(excel_file, conf, add_sections=False):
    # Open Excel file
    xls = pd.ExcelFile(excel_file)
    dfs = {}
    # Read all worksheets
    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name,
                               index_col=None,
                               header=None,
                               keep_default_na=False,
                               names=COLUMN_NAMES,
                               usecols=[i for i in range(len(COLUMN_NAMES))],
                               skiprows=[i for i in range(HEADER_HEIGHT)])
            # Do NOT remove empty lines in order to keep line numbers as in the original file
            dfs[excel_file + '_' + sheet_name] = df
        except ValueError:
            logging.warning(f'Ignored sheet {sheet_name} in file {excel_file}: unexpected format')

    return parse_schema(dfs, conf, add_sections=add_sections)


def parse_schema(dfs, conf, color=None, level=0, add_sections=False):
    all_ids = dict()
    all_paths = set()
    path_to_id = dict()
    slug_to_real_id = dict()
    all_entity_paths = set()
    entities_level1 = []
    brat_entity_children = {}
    attributes = {}
    is_a_hierarchy = dict()  # all is_a relations
    brat_disabled_entities = set()

    if add_sections:
        section_df = build_section_df(conf)
        dfs['sections'] = section_df

    for sheet_name, df in dfs.items():
        for index, row in df.iterrows():
            row_id = row[ID].strip()
            # if not row id, skip the line
            if row_id == '':
                continue

            # Check id uniqueness
            slugified_id = slugify(row_id)
            if slugified_id in slug_to_real_id:
                raise ValueError(error_message(sheet_name,
                                               index,
                                               'id {} has the same slugified value '
                                               '({}) than {}, you should give a different name'.format(
                                                   row_id,
                                                   slug_to_real_id[slugified_id], slugified_id)))
            slug_to_real_id[slugified_id] = row_id
            row_path = row[PATH].strip()
            # if no row path is specified, use the row id
            if row_path == '':
                row_path = row_id
            if row_path in all_paths:
                raise ValueError(error_message(sheet_name, index, 'Duplicate path {}'.format(row_path)))
            all_paths.add(row_path)
            path_to_id[row_path] = slugified_id
            # Brat level -> ignore
            brat_level = get_brat_level(row)
            if brat_level == 0:
                raise ValueError(error_message(sheet_name,
                                               index, 'Brat hierarchy level can be < 0 or > 0, but not == 0'))
            # disabled?
            brat_disabled = DISABLED_CHOICES[row[BRAT_ENABLED]]
            # Build path
            # 1. check attribute
            attribute_split = row_path.split(conf.ATTRIBUTE_SEPARATOR)
            # No attribute in path
            if len(attribute_split) == 1:
                parent_path, brat_parent_path, entity_name = get_parent_path(row_path, sheet_name, index, conf,
                                                                             brat_level)
                # Check that the parent path is valid (existing entities)
                # + update is_a hierarchy
                ent_hierarchy = []
                for i in range(len(parent_path)):
                    parent_id = conf.PATH_SEPARATOR.join(parent_path[:i + 1])
                    if parent_id not in all_paths:
                        raise ValueError(error_message(sheet_name, index,
                                                       'Path refers to unknown entity {}'.format(parent_id)))
                    ent_hierarchy.append(path_to_id[parent_id])
                is_a_hierarchy[slugified_id] = ent_hierarchy
                brat_type = row[BRAT_TYPE].strip()

                # path describes an entity
                if brat_type == ENTITY:
                    if row_path in all_entity_paths:
                        raise ValueError(error_message(sheet_name, index, 'Duplicate entity {}'.format(row_path)))
                    all_entity_paths.add(row_path)
                    # Disabled?
                    if brat_disabled:
                        brat_disabled_entities.add(slugified_id)

                    # length 1
                    if brat_parent_path is None:
                        entities_level1.append(slugified_id)
                        brat_entity_children[slugified_id] = []
                    else:
                        s_brat_parent_path = path_to_id[brat_parent_path]
                        children = brat_entity_children.get(s_brat_parent_path, [])
                        children.append(slugified_id)
                        brat_entity_children[s_brat_parent_path] = children
                    all_ids[slugified_id] = {"type": ENTITY}
                # path describes an attribute
                elif brat_type == ATTRIBUTE:
                    # ignore brat level
                    if brat_level >= 0:
                        logging.warning(error_message(sheet_name,
                                                      index,
                                                      'Ignored level {} for attribute {} '.format(brat_level,
                                                                                                  entity_name)))
                    # A disabled attribute will not appear at all
                    if brat_disabled:
                        logging.info(error_message(sheet_name, index,
                                                   'Attribute {} is disabled, '
                                                   'it will not appear at all'.format(entity_name)))
                        all_ids[slugified_id] = {"type": ATTRIBUTE}
                        brat_disabled_entities.add(slugified_id)
                    else:
                        # add attribute
                        p_id = path_to_id[brat_parent_path]
                        if p_id in brat_disabled_entities:
                            brat_disabled_entities.remove(p_id)

                        add_attribute(sheet_name, index, slugified_id, row_path, None,
                                      entity_name, attributes, all_ids, all_entity_paths, 
                                      path_to_id, conf)
                # path describes an event
                elif brat_type == EVENT:
                    raise NotImplementedError()
                else:
                    raise SyntaxError(error_message(sheet_name, index, 'Unknown Brat type {}'.format(row[BRAT_TYPE])))
            # Attribute
            elif len(attribute_split) == 2:
                # Type must be attribute
                if row[BRAT_TYPE] != ATTRIBUTE:
                    raise SyntaxError(error_message(sheet_name,
                                                    index,
                                                    'a path with attribute-separator "{}" '
                                                    'must have type {}'.format(conf.ATTRIBUTE_SEPARATOR, ATTRIBUTE)))
                entity_path, attribute_path = attribute_split[0], attribute_split[1]
                
                # attribute path must be key:value
                attribute_path_split = attribute_path.split(conf.KEY_VALUE_SEPARATOR)
                # binary attribute
                if len(attribute_path_split) == 1:
                    key, value = attribute_path_split[0], None
                elif len(attribute_path_split) == 2:
                    key, value = attribute_path_split[0], attribute_path_split[1]
                else:
                    raise SyntaxError(error_message(sheet_name,
                                                    index,
                                                    'attribute must have the form "key" '
                                                    '(binary attributes) or "key{}value"'.format(
                                                        conf.KEY_VALUE_SEPARATOR)))

                # Brat level -> ignore
                if brat_level >= 0:
                    logging.warning(error_message(sheet_name, index,
                                                  'Ignored level {} for attribute {}:{} '.format(brat_level,
                                                                                                 key, value)))
                # A disabled attribute will not appear at all
                if brat_disabled:
                    logging.info(error_message(sheet_name, index,
                                               'Attribute {}:{} is disabled, it will not appear at all'.format(key,
                                                                                                               value)))
                else:
                    add_attribute(sheet_name, index, slugified_id, entity_path, key, value, attributes,
                                  all_ids, all_entity_paths, path_to_id, conf)

            # Syntax error
            else:
                raise SyntaxError(error_message(sheet_name, index,
                                                'cannot have more than 1 "{}" in a line '
                                                '(only one attribute)'.format(conf.ATTRIBUTE_SEPARATOR)))

    return all_ids, slug_to_real_id, is_a_hierarchy, entities_level1, \
           brat_entity_children, brat_disabled_entities, attributes
