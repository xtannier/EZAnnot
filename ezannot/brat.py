import re
from os.path import isfile, join
from .toolbox import slugify, get_colors

BRAT_NEWLINE_REGEX = re.compile('[^\n]*[^\s]')


def to_brat_entity(entity_index, entity_type, start, end, text):
    texts = []
    offsets = []
    for i, match in enumerate(BRAT_NEWLINE_REGEX.finditer(text)):
        match_start = start + match.start()
        match_end = start + match.end()
        texts.append(text[match_start - start:match_end - start])
        offsets.append('{} {}'.format(match_start, match_end))
    if len(texts) == 0:
        texts.append(text[start:end])
        offsets.append('{} {}'.format(start, end))

    assert len(texts) > 0, '{} - {} - {}'.format(text, start, end)
    return 'T{}\t{} {}\t{}\n'.format(entity_index, entity_type, ';'.join(offsets), ' '.join(texts))


def should_preannotate(ann_file, conf, overwrite_ann=False):
    # if .ann exists
    if isfile(ann_file):
        if not overwrite_ann:
            return False
        # If option overwrite .ann
        # check that the manual annotation has not been achieved
        # (entity metadata, attribute done)
        else:
            with open(ann_file, 'r', encoding='utf-8') as brat_ann:
                att_checked = None
                entity_checked = None
                for line in brat_ann:
                    match = conf.BRAT_DONE_REGEX.match(line)
                    if match is not None:
                        att_checked = match.group(1)
                        if entity_checked is not None and entity_checked == att_checked:
                            return False
                    else:
                        match = conf.BRAT_METADATA_REGEX.match(line)
                        if match is not None:
                            entity_checked = match.group(1)
                            if att_checked is not None and entity_checked == att_checked:
                                return False
            return True
    else:
        return True


def write_entity(entities, entity_children, disabled_entities, attributes, out, level=0):
    for s_entity in entities:
        skip = False
        prefix = ''
        if s_entity in disabled_entities:
            if len(entity_children.get(s_entity, [])) == 0:
                if not any([s_entity in values.keys() for values in [v for k, v in attributes.items()]]):
                    skip = True
            else:
                prefix = '!'
        if not skip:
            out.write('\t' * level + prefix + s_entity + '\n')
        write_entity(entity_children.get(s_entity, []), entity_children, disabled_entities, attributes, out,
                     level=level + 1)


def write_label(entities, entity_children, out, conf):
    for elem_name in entities:
        out.write(slugify(elem_name))
        if conf.ALLOW_SHORT_NAMES:
            words = elem_name.split(' ')
            for i in range(len(words), 0, -1):
                out.write(' | ' + ' '.join(words[:i]))
        else:
            out.write(' | ' + elem_name)
        out.write('\n')

        write_label(entity_children.get(elem_name, []), entity_children, out, conf)


def write_color(entities, entity_children, out, color):
    for entity in entities:
        out.write(slugify(entity) + '\tbgColor:' + color + '\n')
        write_color(entity_children.get(entity, []), entity_children, out, color)


def save_brat_schema(ann_dir,
                     brat_entities_level1, brat_entity_children, brat_disabled_entities, brat_attributes,
                     conf,
                     include_visual=True):
    out_annotation_file = join(ann_dir, 'annotation.conf')

    with open(out_annotation_file, 'w') as out:
        out.write('[entities]\n')
        write_entity(brat_entities_level1, brat_entity_children, brat_disabled_entities, brat_attributes, out)
        # add metadata entity
        out.write(conf.METADATA_ENTITY_NAME + '\n')
        out.write('\n\n')
        out.write('[events]\n')
        out.write('\n\n')
        out.write('[attributes]\n')
        out.write(conf.METADATA_ANNOTATION_DONE_ATTRIBUTE + '\tArg:' + conf.METADATA_ENTITY_NAME + '\n')
        for att_k, att_info in brat_attributes.items():
            out.write(slugify(att_k) + '\tArg:' + '|'.join(map(slugify, att_info.keys())))
            # we already checked that all entities associated with this attribute
            # have the same value set. That's why we can just get the first value set.
            att_values = [slugify(a) for a in att_info[list(att_info.keys())[0]] if a is not None]
            if len(att_values) > 0:
                out.write(', Value:' + '|'.join(att_values))
            out.write('\n')
        out.write('\n\n')
        out.write('[relations]\n')
        out.write('<OVERLAP>	Arg1:<ENTITY>, Arg2:<ENTITY>, <OVL-TYPE>:<ANY>\n')
        out.write('\n\n')

    if include_visual:
        out_visual_conf = join(ann_dir, 'visual.conf')

        with open(out_visual_conf, 'w') as out:
            out.write('[labels]\n')
            write_label(brat_entities_level1, brat_entity_children, out, conf)
            out.write('\n\n')
            out.write('[drawing]\n')
            colors = list(get_colors(len(brat_entities_level1)))
            for elem_name, color in zip(brat_entities_level1, colors):
                out.write(slugify(elem_name) + '\tbgColor:' + color + '\n')
                write_color(brat_entity_children.get(elem_name, []), brat_entity_children, out, color)
    else:
        out_visual_conf = None

    return out_annotation_file, out_visual_conf
