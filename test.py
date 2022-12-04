#%%
import urllib
import bs4
import pandas
from urllib import request
import requests

#%%
url = 'http://sofifa.com/players?offset=0'

# %%
# Fonction génératrice
def soup_maker(url):
    r = requests.get(url)
    markup = r.content
    soup = bs4.BeautifulSoup(markup, 'lxml')
    return soup

#%%
#Fonction donnant la liste des url de chaque joueur
def all_ref(url):
    L=[]
    soup=soup_maker(url)
    table = soup.find('table', {'class': 'table-hover'})
    tbody = table.find('tbody', {'class': 'list'})
    col_name = tbody.find_all('tr')
    for col in col_name:
        col_ref = col.find('a')
        L.append(col_ref['href'])
    return ["http://sofifa.com" + L[i] for i in range(len(L))]


# %%
#Fonction donnant le nom d'un joueur à partir de son url
def name (url):
    soup = soup_maker(url)
    info=soup.find('div', {'class': "col-12"})
    return info.find('h1').text.strip()

# %%
