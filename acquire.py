import requests
from bs4 import BeautifulSoup 
import pandas as pd
import re

def get_blog_articles(url):
    '''
    can take in a list of urls and returns a dictionary that can be turned into a pandas dataframe.
    '''
    response = requests.get(f'{url}',headers = {'user-agent': 'Codeup DS Germain'})
    html = response.text
    soup = BeautifulSoup(html)
    article = soup.select('.et_pb_row.et_pb_row_0_tb_body')
    article = article[0]
    title = article.h1.text
    contents = article.select('p')[1:]
    content_holder = ''
    
    for content in contents:
        content_holder += content.text + ""
    return {'title':title,'content':content_holder}