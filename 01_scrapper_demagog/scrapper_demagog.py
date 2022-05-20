from bs4 import BeautifulSoup
import requests

import time
import numpy as np
import pandas as pd

from tqdm import tqdm

import unicodedata

demagog_statements_url = 'https://demagog.org.pl/wypowiedzi/page/'
page = requests.get(f'{demagog_statements_url}1')

soup = BeautifulSoup(page.content, 'html.parser')

n_pages = int(soup.findAll("a", class_="page-numbers")[1].text)

# if False:
#     df_in = pd.read_csv('../datasets/demagog.csv', sep=';')
#     txt_list = df_in['text'].values.tolist()
#     assestment_list = df_in['assestment'].values.tolist()
# else:
#     txt_list = []
#     assestment_list = []

txt_list = []
assestment_list = []
author_list = []
source_list = []

list_wrong_url = []

for i in range(1,n_pages+1):
    
    page=requests.get(f'{demagog_statements_url}{i}')
    soup = BeautifulSoup(page.content, 'html.parser')
    
    for s in tqdm(soup.findAll("h2", class_="mt-0 mb-1 title-archive"), 
                  desc=f'page {i} of {n_pages}', 
                  position=0
                 ):
        statement_url = s.find('a').get("href")
        
        try:
            page_statement = requests.get(statement_url)
            soup_statement = BeautifulSoup(page_statement.content, 'html.parser')

            txt = unicodedata.normalize("NFKD", 
                                        soup_statement.find('blockquote', class_='hyphenate target-blank twitter-tweet') \
                    .find('p').text)

            assestment = unicodedata.normalize("NFKD", 
                                               soup_statement.find_all('p', {"class": "ocena"})[0].text)
            author = unicodedata.normalize("NFKD", 
                                           soup_statement.find('div', class_='h2 person-name mt-0 count-text').find('a').text)
            if soup_statement.find('div', class_='w-100 text-right date-content target-blank'):
            	source = unicodedata.normalize("NFKD", 
                                           soup_statement.find('div', class_='w-100 text-right date-content target-blank').find('p').text)
            else:
            	source = ['wypowiedź']

            txt_list.append(txt)
            assestment_list.append(assestment)
            author_list.append(author)
            source_list.append(source)
        
            #time.sleep(0.1)
            
        except Exception as e: 
            print(e)
            print(statement_url)    
            list_wrong_url.append(statement_url)


    df_out = pd.DataFrame({'assestment' : assestment_list, 'text' : txt_list, 'author' : author_list, 'source' : source_list})
    df_out.to_csv('../datasets/demagog.csv', sep=';', index=False)
    
    
df_out = pd.DataFrame({'assestment' : assestment_list, 'text' : txt_list, 'author' : author_list, 'source' : source_list})
df_out.to_csv('../datasets/demagog.csv', sep=';', index=False)
