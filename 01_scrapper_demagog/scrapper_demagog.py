from bs4 import BeautifulSoup
import requests

import time
import numpy as np

from tqdm import tqdm

demagog_statements_url = 'https://demagog.org.pl/wypowiedzi/page/'
page = requests.get(f'{demagog_statements_url}1')

soup = BeautifulSoup(page.content, 'html.parser')

n_pages = int(soup.findAll("a", class_="page-numbers")[1].text)

txt_list = []
assestment_list = []
txt_list = []
assestment_list = []
for i in range(1,n_pages+1):
    
    page=requests.get(f'{demagog_statements_url}{i}')
    soup = BeautifulSoup(page.content, 'html.parser')
    
    for s in tqdm(soup.findAll("h2", class_="mt-0 mb-1 title-archive"), 
                  desc=f'page {i} of {n_pages}', 
                  position=0
                 ):
        statement_url = s.find('a').get("href")

        page_statement = requests.get(statement_url)
        soup_statement = BeautifulSoup(page_statement.content, 'html.parser')

        txt = soup_statement.find('blockquote', class_='hyphenate target-blank twitter-tweet') \
                .find('p').text

        assestment = soup_statement.find_all('p', {"class": "ocena"})[0].text
        
        txt_list.append(txt)
        assestment_list.append(assestment)
        
        time.sleep(np.random.randint(5)/5)
    
df_out = pd.DataFrame({'assestment' : assestment_list, 'text' : txt_list})
df_out.to_csv('../datasets/demagog.csv', sep=';', index=False)