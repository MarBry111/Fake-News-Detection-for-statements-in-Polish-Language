import pandas as pd
import numpy as np
import spacy
from sentimentpl.models import SentimentPLModel
from autocorrect import Speller

import re
import unicodedata

import gensim

from tqdm import tqdm

###############################
######## preprocessing ########
###############################
def deal_with_polish_sign(txt):
    
    txt = (txt.replace('\n', ' ')
       .replace('ą', 'ą')
       .replace('ć', 'ć')
       .replace('ę', 'ę')
       .replace('ń', 'ń')
       .replace('ó', 'ó')
       .replace('ś', 'ś')
       .replace('ź', 'ź')
       .replace('ż', 'ż'))
    
    return txt


def tokenize(txt, nlp_core, stopwords, join_str=None):
    """
    Function to tokenize pice of text applying:
    - repalcement of strange chars (polish language)
    - keepign not stop words, punct, like number, spaces, words shorter than 3 chars
    """
    txt =  re.sub(' +', ' ', txt)

    txt = deal_with_polish_sign(txt)

    doc = nlp_core(txt)
    
    words = [
        token.lemma_.lower()
        for token in doc 
        if 
            not token.is_stop 
            and not token.is_punct 
            and not token.like_num
            and token.text != ' '
            and token.lemma_ not in stopwords
            and len(token.text) > 2 ]

    if join_str:
        words = join_str.join(words)
    
    return words

def tokenize_pos(txt, nlp_core, join_str=None):
    
    doc = nlp_core(txt)

    txt_pos = []
    for token in doc:
        if token.pos_ not in ['SPACE', 'PUNCT']:
            txt_pos.append(token.pos_)            
    
    if join_str:
        txt_pos = join_str.join(txt_pos)
        
    return txt_pos


def filter_stop_words(words, stop_words):
    """
    functon to filet extra stop words
    """
    out = [x for x in words if x not in stop_words]
    return out


def clean_text(txt):
    """
    function to clean input text
    - clean parts of demagog additions
    - not used braces
    - remove @ sign
    - remove http links
    - deal with html encoding
    - replace ; with . and double spaces with single one
    """
    if txt != txt:
        return np.nan
    
    txt_out = txt
    
    if "przyp. Demagog" in txt_out:
        txt_out = (txt_out
                   .replace('(','').replace(')','')
                   .replace(' – przyp. Demagog','')
                   .replace('- red.', ''))
    if "(…)" in txt_out:
        txt_out =  txt_out.replace('(…)','')
    if "(...)" in txt_out:
        txt_out =  txt_out.replace('(...)','')
    if "[" in txt_out:
        txt_out = txt_out.replace('[','').replace(']','')
        
    txt_out = re.sub("@[A-Za-z0-9]+","",txt_out) #Remove @ sign
    txt_out = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", txt_out) #Remove http links
    
    txt_out = unicodedata.normalize("NFKD", txt_out) #cleaning html
    
    txt_out = txt_out.replace(';', '.').replace('  ', ' ')
    
    return txt_out


###############################
#### stylommetric features ####
###############################
def get_vowels_per_word(t):
    vowels = [len(re.findall('[aeiouóyąę]+', w)) for w in re.findall('(?![\d\s])[\w]+(?![\d\s])', t)]
    vowels = np.array(vowels)
    vowels = vowels[vowels>0]
    
    if vowels.shape[0] == 0:
        return [0]
    return vowels

def get_vocab_rich_features(txt, nlp_core):
    uniq_words = {}
    words = ' '.join(re.findall('(?![\d])[\w]+', txt)).strip()
    doc = nlp_core(words)
    
    for token in doc:
        if token.pos_ not in ['SPACE', 'PUNCT', 'SYM', 'X', 'NUM']:
            lemma = token.lemma_
            if lemma in uniq_words:
                uniq_words[lemma] = uniq_words[lemma] + 1
            else:
                uniq_words[lemma] = 1
    
    n = len(words.split(' '))
    v = len(uniq_words)
    
    # Hapax Legomena and Hapax DisLegemena
    v1 = sum(x == 1 for x in uniq_words.values())
    v2 = sum(x == 2 for x in uniq_words.values())
    vi = {}
    for k in uniq_words.keys():
        n_w = uniq_words[k]
        if n_w in vi:
            vi[n_w] = vi[n_w] + 1
        else:
            vi[n_w] = 1
    # Honore’s measure R
    R = 100 * np.log(n+1) / (1 - v1/v + 1)
    # Sichel’s measure S 
    S = v2/v
    # Brunet’s measure W 
    # https://linguistics.stackexchange.com/questions/27365/formula-for-brun%C3%A9ts-index
    W = n**(v**(-0.17))
    # Yule’s characteristic K
    M = np.sum([n_w**2 * vi[n_w] for n_w in vi])
    K = 10**4 * (M-n)/(n**2)
    # Shannon Entropy
    E = np.sum([uniq_words[w]/n * np.log(uniq_words[w]/n) for w in uniq_words])
    # Simpson’s index D
    D = np.sum([(uniq_words[w]/n)**2 for w in uniq_words])
    # type token ratio (TTR)
    T = v/n
    
    return v1, v2, R, S, W, K, E, D, T


def get_sentiment(txt, nlp_core, model_sent):
    doc = nlp_core(txt)
    
    sentiment_all = model_sent(txt).item()
    
    sentiments_list = []
    for i, sent in enumerate(doc.sents):
        s = model_sent(sent.text).item()
        sentiments_list.append(s)
    
    sentiment_avg = np.mean(sentiments_list)
    
    return sentiment_all, sentiment_avg


def get_vowels_per_word_complex(t, c=2):
    vowels = [len(re.findall('[aeiouóyąę]+', w)) for w in re.findall('(?![\d\s])[\w]+(?![\d\s])', t)]
    vowels = np.array(vowels)
    # complex word - more than 2 syllabes
    vowels = vowels[vowels>c]
    
    if vowels.shape[0] == 0:
        return [0]
    return vowels


def get_n_stop_words(txt, nlp_core, stopwords):
    doc = nlp_core(txt)
    
    n_sw = 0
    
    for token in doc:
        if token.lemma_ in stopwords:
            n_sw = n_sw + 1
    
    return n_sw


def get_pos(txt, nlp_core):
    
    doc = nlp_core(txt)

    adj_n = 0
    adv_n = 0
    noun_n = 0
    ent_n = 0
   
    for token in doc:
        if token.pos_ == 'ADJ': 
            adj_n += 1
        elif token.pos_ == 'ADV':
            adv_n += 1
        elif token.pos_ == 'NOUN':
            noun_n += 1    

    for ent in doc.ents:
        ent_n += 1

    return (
        ent_n, 
        adj_n/len(txt.split(' ')), adj_n, 
        adv_n/len(txt.split(' ')), adv_n, 
        noun_n/len(txt.split(' ')), noun_n,
    )


def get_stylometric_features(df, nlp_core, model_sent, stopwords, text_clean_column='text_clean', join_str=" ", rerun_all=True):
    # Get lexical features
    if rerun_all:
        print('## Get lexical features ##')
        df['avg_word_len'] = df[text_clean_column].progress_apply(
            lambda x: np.mean(
                [ len(w.strip()) for w in re.findall('(?![\d])[\w]+', x)]
            )
        )
    
        df['n_words'] = df[text_clean_column].progress_apply(
            lambda x: len( re.findall('(?![\d])[\w]+', x) )
        )
    
        df['n_char'] = df[text_clean_column].progress_apply(
            lambda x: len(x)
        )
    
        df['n_special_char'] = df[text_clean_column].progress_apply(
            lambda x: len(re.findall('(?![\d\s])[\W]', x))
        )
    
        df['avg_n_vowels_per_word'] = df[text_clean_column].progress_apply(
            lambda x: np.mean(get_vowels_per_word(x.lower()))
        )
        
        ## Vocab richness
        print('## Vocab richness ##')
        vocab_rich_f = df[text_clean_column].progress_apply(
            lambda x: get_vocab_rich_features(x, nlp_core)
        )
        df[
            ['hapax_legomena',
             'hapax_dislegemena',
             'honore_r',
             'sichel_s',
             'brunet_w',
             'yule_k',
             'shannon_entropy',
             'simpson_idx_d',
             'type_token_ratio'
            ]
        ] = vocab_rich_f.values.tolist()
        
        ## Readability
        print('## Readability ##')
        df['FR_score'] = df[text_clean_column].progress_apply(
            lambda x: 
            206.835 
            - 1.015 * len( re.findall('(?![\d])[\w]+', x) ) #total words
            - 84.6 *  np.sum(get_vowels_per_word(x.lower())) / len( re.findall('(?![\d])[\w]+', x) ) #total syllabes/ total words
        )
    
        df['FKG_level'] = df[text_clean_column].progress_apply(
            lambda x: 
            0.39 * len( re.findall('(?![\d])[\w]+', x) ) #total words
            + 11.8 * np.sum(get_vowels_per_word(x.lower())) / len( re.findall('(?![\d])[\w]+', x) ) #total syllabes/ total words
            - 15.59
        )
    
        df['Gunning_Fog_index'] = df[text_clean_column].progress_apply(
            lambda x: 
            0.4 * (
                len( re.findall('(?![\d])[\w]+', x) ) #total words
                + 100 * len(get_vowels_per_word_complex(x.lower())) / len( re.findall('(?![\d])[\w]+', x) ) 
            ) 
        )
        
        ## Add Sentiment
        print('## Add Sentiment ##')
        sentiment_f = df[text_clean_column].progress_apply(
            lambda x: get_sentiment(x, nlp_core, model_sent)
        )
        df[
            ['sentiment_all',
             'sentiment_avg'
            ]
        ] = sentiment_f.values.tolist()
        
        ## Extra features
        print('## Extra features ##')
        
        df['n_stop_words'] = df[text_clean_column].progress_apply(
            lambda x: get_n_stop_words(x, nlp_core, stopwords)
        )
    
        
        pos_f = df[text_clean_column].progress_apply(
            lambda x: get_pos(x, nlp_core)
        )
    
        df[
            ['n_ent',
             'p_adj',
             'n_adj',
             'p_adv',
             'n_adv',
             'p_noun',
             'n_noun',
            ]
        ] = pos_f.values.tolist()

    print('## Add WORDS ##')
    df['TEXT_WORD'] = df[text_clean_column].progress_apply(
        lambda x: tokenize(x, nlp_core, stopwords, join_str)
    )

    print('## Add POS ##')
    df['TEXT_POS'] = df[text_clean_column].progress_apply(
        lambda x: tokenize_pos(x, nlp_core, join_str)
    )

    return df