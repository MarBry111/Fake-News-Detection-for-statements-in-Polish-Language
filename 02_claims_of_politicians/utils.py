import pandas as pd
import numpy as np
import spacy
from sentimentpl.models import SentimentPLModel
from autocorrect import Speller

import re
import unicodedata

# import polyglot
# from polyglot.text import Text, Word

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer#for word embedding

# import gensim
# from gensim.models import Word2Vec

# from tqdm import tqdm
# tqdm.pandas()

# from pandarallel import pandarallel
# pandarallel.initialize(nb_workers=7,progress_bar=True)

# from parallelbar import progress_map

nlp_core = spacy.load("pl_core_news_lg")
model = SentimentPLModel(from_pretrained='latest')
spell = Speller('pl')

stopwords = nlp_core.Defaults.stop_words

def tokenize(txt, nlp_core=nlp_core, stopwords=stopwords):
    txt = (txt.replace('\n', ' ')
           .replace('ą', 'ą')
           .replace('ć', 'ć')
           .replace('ę', 'ę')
           .replace('ń', 'ń')
           .replace('ó', 'ó')
           .replace('ś', 'ś')
           .replace('ź', 'ź')
           .replace('ż', 'ż')
           .replace('  ', ' '))

    doc = nlp_core(txt)
    
    words = [
        token.lemma_.lower()
        for token in doc 
        if 
            not token.is_stop 
            and not token.is_punct 
            and not token.is_stop 
            and not token.is_digit
            and token.text != ' '
            and token.lemma_ not in stopwords
            and len(token.text) > 2 ]
    
    return words


# text cleaning
def clean_przyp(txt):
    if txt != txt:
        return np.nan
    
    txt_out = txt
    
    if "przyp. Demagog" in txt:
        txt_out = (txt_out
                   .replace('(','').replace(')','')
                   .replace(' – przyp. Demagog','')
                   .replace('- red.', ''))
    if "(…)" in txt:
        txt_out =  txt_out.replace('(…)','')
    if "(...)" in txt:
        txt_out =  txt_out.replace('(...)','')
    if "[" in txt:
        txt_out = txt_out.replace('[','').replace(']','')
        
    txt_out = re.sub("@[A-Za-z0-9]+","",txt_out) #Remove @ sign
    txt_out = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", txt_out) #Remove http links
    
    txt_out = unicodedata.normalize("NFKD", txt_out) #cleaning html
    
    txt_out = txt_out.replace(';', '.').replace('  ', ' ')
    
    return txt_out

#extract features
def extract_features(txt, nlp_core=nlp_core):
    
    doc = nlp_core(txt)
    
    out_dict = {}
    
    lemmas_list = []
    tokens_list = []
    sentiments_list = []
    embeddings_list = []

    error_n = 0

    adj_n = 0
    adv_n = 0
    noun_n = 0
    ent_n = 0
    
    upper_s_n = 0
    upper_f_n = sum(map(str.isupper, txt.split()))
    
    e_mark_n = txt.count('!')
    q_mark_n = txt.count('?')
    
    upper_n = sum(1 for c in txt if c.isupper())
    
    out_dict['sentiment_all'] = model(doc.text).item()
    
    for i, sent in enumerate(doc.sents):
        s = model(sent.text).item()
        sentiments_list.append(s)
    
    out_dict['sentiment_avg'] = np.mean(sentiments_list)

    txt_pos = []
    txt_word = []
    for token in doc:
        if token.pos_ not in ['SPACE', 'PUNCT']:
            lemmas_list.append(token.lemma_)
            tokens_list.append(token.text)
            corrected = spell(token.text)
            if corrected != token.text:
                error_n += 1

        # if token.pos_ == 'ADJ': 
        #     adj_n += 1
        # elif token.pos_ == 'ADV':
        #     adv_n += 1
        # elif token.pos_ == 'NOUN':
        #     noun_n += 1
        
            if token.text[0].isupper():
                upper_s_n += 1
            
            txt_pos.append(token.pos_)            

    for ent in doc.ents:
        ent_n += 1

    tokens_list = list(set(tokens_list))
    lemmas_list = list(set(lemmas_list))

    out_dict['uniq_words'] = len(tokens_list)
    out_dict['uniq_lemm'] =  len(lemmas_list)
    out_dict['err'] =  error_n
    out_dict['net'] = ent_n
    # out_dict['ADJ'] = adj_n/len(tokens_list)
    # out_dict['ADV'] = adv_n/len(tokens_list)
    # out_dict['NOUN'] = noun_n/len(tokens_list)
    
    out_dict['words_start_upper'] = upper_s_n
    out_dict['words_full_upper'] = upper_f_n
    out_dict['exclamation_marks'] = e_mark_n
    out_dict['question_marks'] = q_mark_n
    
    out_dict['upper_letters'] = upper_n
    out_dict['chars'] = len(txt)

    out_dict['TEXT_POS'] = txt_pos

    out_dict['TEXT_WORD'] = tokenize(txt)
    
    return out_dict