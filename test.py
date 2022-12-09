#%%
import urllib
import bs4
import pandas
from urllib import request
import requests

#%%
url = 'http://sofifa.com/players?offset=0'

# %%
# Fonction génératrice de page html
def soup_maker(url):
    r = requests.get(url)
    markup = r.content
    soup = bs4.BeautifulSoup(markup, 'lxml')
    return soup

#%%
#Fonction donnant la liste des url de chaque n*60 joueur
def all_ref(n):
    L=[]
    for i in range(n):
        url='http://sofifa.com/players?offset=' + str(i*60)
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
#Fonction donnant l'âge d'un joueur
def age (url):
    soup = soup_maker(url)
    info=soup.find('div', {'class': "col-12"})
    bio = info.find('div', {'class': 'meta ellipsis'})
    bio_str = bio.text.strip()
    bio = bio_str.split()
    return bio[-6]

#%%
#Fonction donnant le poids d'un joueur
def poids(url):
    soup = soup_maker(url)
    info=soup.find('div', {'class': "col-12"})
    bio = info.find('div', {'class': 'meta ellipsis'})
    bio_str = bio.text.strip()
    bio = bio_str.split()
    return bio[-1]

#%%
#Fonction donnant la taille d'un joueur
def taille(url):
    soup = soup_maker(url)
    info=soup.find('div', {'class': "col-12"})
    bio = info.find('div', {'class': 'meta ellipsis'})
    bio_str = bio.text.strip()
    bio = bio_str.split()
    return bio[-2]

# %%
# Fonction donnant overall_rating d'un joueur
def overall_rating(url):
    soup=soup_maker(url)
    info=soup.find('div', {'class': "bp3-card"})
    info_stat=info.find('section', {'class': 'card spacing'})
    all=info_stat.findAll('div', {'class': 'block-quarter'})
    a=all[0].find('span')
    return a.text.strip()

#%% 
# Fonction donnant potential d'un joueur
def potential(url):
    soup=soup_maker(url)
    info=soup.find('div', {'class': "bp3-card"})
    info_stat=info.find('section', {'class': 'card spacing'})
    all=info_stat.findAll('div', {'class': 'block-quarter'})
    a=all[1].find('span')
    return a.text.strip()

#%% 
# Fonction donnant la valeur d'un joueur
def value(url):
    soup=soup_maker(url)
    info=soup.find('div', {'class': "bp3-card"})
    info_stat=info.find('section', {'class': 'card spacing'})
    all=info_stat.findAll('div', {'class': 'block-quarter'})
    a=all[2].find('div')
    b=a.text.strip()
    return b.split('V')[0]

#%% 
# Fonction donnant le salaire d'un joueur
def wage(url):
    soup=soup_maker(url)
    info=soup.find('div', {'class': "bp3-card"})
    info_stat=info.find('section', {'class': 'card spacing'})
    all=info_stat.findAll('div', {'class': 'block-quarter'})
    a=all[3].find('div')
    b=a.text.strip()
    return b.split('W')[0]

#%%
# Fonction donnant la valeur de la clause de rupture
def release_clause(url):
    soup=soup_maker(url)
    info=soup.find('ul', {'class': 'pl'})
    info_sup=info.findAll('li', {'class': 'ellipsis'})
    a=info_sup[-2].find('span')
    return a.text.strip()

#%%
# Fonction donnant la position de jeu d'un joueur
def position(url):
    oup=soup_maker(url)
    info=soup.find('ul', {'class': 'ellipsis pl'})
    info_pos=info.findAll('li')
    a=info_pos[-4].find('span')
    return a.text.strip()

#%%
url_1='https://sofifa.com/player/178509/olivier-giroud/230006/'
soup=soup_maker(url_1)
info=soup.find('div', {'class': "bp3-card"})

#%%
url_2='https://sofifa.com/player/239837/alexis-mac-allister/230007/'

#%%
soup2=soup_maker(url_2)