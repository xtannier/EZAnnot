---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# EZAnnot example

See `README.md` for the documentation on resources.

```python
# autoreload external modules
%load_ext autoreload
%autoreload 2

import sys
from os import makedirs, listdir, remove
from os.path import isfile, isdir, exists, join, basename
from tqdm.auto import tqdm
import re
import pandas as pd
import logging

import spacy

# sys.path.append('../ezannot')   # Add here your path to EZAnnot if it is not installed
                                  # your main directory

from ezannot import searcher, schema, brat, rules, custom_boolean
from ezannot import configuration_sys as conf_sys
from ezannot.schema import parse_schema_excel
from ezannot.toolbox import slugify, length_preserving_clean_text

# Import configuration.py, containing your custom configuration
# Generally you won't need to change anything
import configuration
conf = configuration.Configuration()
```

## Inputs and outputs

```python
# Logging configuration
logging.basicConfig(stream=sys.stdout, 
                    format='%(asctime)s : %(levelname)s : %(message)s', 
                    level=logging.INFO)

# Directory containing the text
# put here for the example, but should be in $HOME/brat_data arborescence in you want a Brat output
in_text_directory = join('example', 'data')

# spaCy model
spacy_model = "/export/home/opt/data/spacy/fr_core_news_sm-2.1.0/fr_core_news_sm/fr_core_news_sm-2.1.0"

# Custom rule funtions
import example.doc.example_custom_rules as c
custom_rule_functions = [c.do_nothing] 

# Schema configuration Excel file
in_excel_schema_configuration = join('example', 'doc', 'example_schema.xlsx')

# Terminology Excel files
in_excel_terminologies = [join('example', 'doc', f) for f in ['example_terminology.xlsx', 'example_terminology_negation.xlsx']]

# Rule Excel files
in_excel_rules = [join('example', 'doc', f) for f in ['example_rules.xlsx']]

# Section splitter resources (None if not used)
# Credits for the section splitter: Ivan Lerner
in_section_splitter_resources = join('example', 'resources', 'section_headings', 'section_normed_v2.pickle')
    
# Write output in Brat and/or dataframes
output_types = ['brat', 'dataframes'] #, 'dataframes']   
assert all(output_type in ['dataframes', 'brat'] for output_type in output_types)
    
# Output directories
out_directories = {}    
if 'brat' in output_types:
    # Brat output directory (the same as the input directory)
    out_directories['brat'] = in_text_directory
    # Overwrite existing .ann if not already manually annotated
    overwrite_ann = True
else:
    overwrite_ann = False
if 'dataframes' in output_types:
    # output directory
    out_directories['dataframes'] = join('example', 'out')

# Sanity checks
for f in in_excel_terminologies + in_excel_rules + [in_excel_schema_configuration, in_section_splitter_resources]:
    if f is not None and not isfile(f):
        logging.error('{} not found'.format(f))
if not isdir(in_text_directory):
    logging.error('{} not found'.format(in_text_directory))
for out_directory in out_directories.values():
    if not isdir(out_directory):
        makedirs(out_directory)
    
rule_parser = custom_boolean.CustomBooleanAlgebra()
```

## Load resources

### Load spaCy model (long!)

```python
logging.info('Load spaCy')
nlp = spacy.load(conf.SPACY_MODEL, disable=['ner', 'tagger', 'parser'])
```

### Parse schema from Excel file

```python
ids, slug_to_real_id, is_a_hierarchy, brat_entities_level1, brat_entity_children, brat_disabled_entities, brat_attributes = parse_schema_excel(
    in_excel_schema_configuration, 
    conf,
    add_sections=True,
)
```

#### Save schema in Brat format (for annotation colors and shortcuts)

```python
if 'brat' in output_types:
    brat.save_brat_schema(
        out_directories['brat'],
        brat_entities_level1, 
        brat_entity_children, 
        brat_disabled_entities, 
        brat_attributes,
        conf
    ) 
    logging.info(f'Brat configuration files saved to ' + out_directories['brat'])
```

#### Add section headers

```python
if conf.SECTION_SPLITTER is not None:
    section_ids = {slugify(conf.SECTION_PREFIX + s) for s in conf.SECTION_TYPES}
else:
    section_ids = set()
```

### Load terminology files

```python
logging.info('Initialize terminology searcher')
terminology_searcher = searcher.TerminologySearcher(conf,
                                            threshold=conf.SIMILARITY_THRESHOLD,
                                            spacy_model_object=nlp, 
                                            section_splitter_path=in_section_splitter_resources,
                                            rule_parser=rule_parser)

entity_names_for_second_level_rules = terminology_searcher.load_terminologies(in_excel_terminologies, ids)
```

### Load rule files

```python
rule_dict = rules.load_rules(in_excel_rules, rule_parser, ids, section_ids, conf)
```

## Launch annotation

```python
def get_paths_to_annotate(in_text_directory, 
                          output_mode,
                          conf,
                          out_brat_data_directory=None, 
                          overwrite_ann=False):
    '''
    Generates a pair (texts, out_ids) where out is either
    - (brat output mode) the paths to the output .ann 
    or
    - (dataframes output mode) the ids of the texts 
    '''
    in_text_files = [f for f in listdir(in_text_directory) if f.endswith('txt')]

    #in_text_files = [f for f in in_text_files if '-1634981805766787313' in f]
    
    if output_mode == 'brat':
        text_paths = []
        out_ids = []
        for text_file in in_text_files:
            text_path = join(out_brat_data_directory, text_file)
            # find .ann
            ann_path = text_path[:-4] + '.ann'
            # Preannotate only if not already annotated by an expert
            if brat.should_preannotate(ann_path, conf, overwrite_ann=overwrite_ann):
                text_paths.append(text_path)
                out_ids.append(ann_path)
            else:
                logging.info('Skip ' + text_file)
    elif output_mode == 'dataframes':
        text_paths = [join(in_text_directory, file) for file in in_text_files]
        out_ids = [basename(path)[:-4] for path in in_text_files]
    else:
        raise ValueError(output_mode)
    return text_paths, out_ids
```

```python
!export NUMEXPR_MAX_THREADS=64
```

```python
debug_level=0
n_process=1

logging.info('Get files to annotate')
# Get all files
in_text_files = [f for f in listdir(in_text_directory) if f.endswith('txt')]

for output_type in output_types:

    text_paths, out_ids = get_paths_to_annotate(in_text_directory, output_type, 
                                                conf, 
                                                out_brat_data_directory=out_directories[output_type], 
                                                overwrite_ann=overwrite_ann)
    logging.info('Parse files')
    if output_type == 'brat':
        ann_paths = out_ids
        terminology_searcher.text_batch_to_brat(text_paths, ann_paths, rule_dict, ids, is_a_hierarchy=is_a_hierarchy, 
                                                second_level_rules=entity_names_for_second_level_rules, 
                                                brat_disabled_entities=brat_disabled_entities, custom_rule_functions=custom_rule_functions,
                                                n_process=n_process, debug_level=debug_level)
        logging.info('Brat annotations written to ' + out_directories['brat'])     
    elif output_type == 'dataframes':
        dataframes = terminology_searcher.text_batch_to_df(text_paths, out_ids, rule_dict, ids, is_a_hierarchy=is_a_hierarchy, 
                                                           second_level_rules=entity_names_for_second_level_rules, 
                                                           brat_disabled_entities=brat_disabled_entities, custom_rule_functions=custom_rule_functions,
                                                           n_process=n_process, debug_level=debug_level)
        df_ehr_phenotyping = dataframes[0]
        df_ehr_phenotyping_ref_type = dataframes[1]
        df_ehr_phenotyping_relations = dataframes[2]
        df_ehr_phenotyping_relation_ref_type = dataframes[3]
    else:
        raise ValueError(output_type)
```

#### Explore dataframes (if `dataframe` output was asked)

```python
if 'dataframes' in output_types:
    display(df_ehr_phenotyping.head())
    display(df_ehr_phenotyping_ref_type.head())
    display(df_ehr_phenotyping_relations.head())
    display(df_ehr_phenotyping_relation_ref_type.head())
```

#### Save dataframes

```python
if 'dataframes' in output_types:
    out_dir_df = out_directories['dataframes']
    df_ehr_phenotyping.to_pickle(join(out_dir_df, 'df_ehr_phenotyping.pkl'))
    df_ehr_phenotyping_ref_type.to_pickle(join(out_dir_df, 'df_ehr_phenotyping_ref_type.pkl'))
    df_ehr_phenotyping_relations.to_pickle(join(out_dir_df, 'df_ehr_phenotyping_relations.pkl'))
    df_ehr_phenotyping_relation_ref_type.to_pickle(join(out_dir_df, 'df_ehr_phenotyping_relation_ref_type.pkl'))
```
