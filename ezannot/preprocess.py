from flashtext import KeywordProcessor
import pickle
import pandas as pd

class SectionSplitter(object):
    '''
    Section splitter from Ivan Lerner
    https://gitlab.eds.aphp.fr/IvanL/clinical_information_extraction/-/blob/master/extract/preprocess.py
    '''
    def __init__(self, 
                 terminology_path:str="",
                 mode="section_v1"):
        
        with open(terminology_path, "rb") as f:
            section_dict = pickle.load( f)
            
        self.path = terminology_path
        
        self.keyword_processor = KeywordProcessor(case_sensitive=True)
        self.keyword_processor.add_keywords_from_dict(section_dict)
        self.head_before_treat = [ "histoire", "evolution"]
        self.mode = mode

            
    def transform_text(self, text):
        match = self.keyword_processor.extract_keywords(text, span_info = True)
        match = pd.DataFrame(match, columns=["match_type", "start", "end"]).sort_values(['start','end'])
        match = (match.append({"match_type": 'head', "start":0}, ignore_index=True)
                 .sort_values('start')
                 .assign(end = lambda x:x.start.shift(-1).fillna(len(text)).astype('int'))
                 .assign(sl = lambda x:x.start - x.end).loc[lambda x:x.sl!=0].drop("sl", axis=1)
                 .reset_index(drop=True)
                )
        
        if self.mode == "section_v2":
            #set any traitement section occuring before histoire or evolution to traitement entree
            index_before_treat = match.loc[lambda x:x.match_type.isin(self.head_before_treat)].index.tolist()
            index_before_treat = min(index_before_treat, default=0)
            match.loc[lambda x:(x.match_type == "traitement")&(x.index < index_before_treat), "match_type"] = "traitement_entree"

        return match
    