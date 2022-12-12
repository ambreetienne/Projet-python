#%%
import urllib
import bs4
import pandas as pd
import numpy as np
from urllib import request
import requests
import xlsxwriter

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
def name (soup):
    info=soup.find('div', {'class': "col-12"})
    return info.find('h1').text.strip()

# %% 
#Fonction donnant l'âge d'un joueur
def age (soup):
    info=soup.find('div', {'class': "col-12"})
    bio = info.find('div', {'class': 'meta ellipsis'})
    bio_str = bio.text.strip()
    bio = bio_str.split()
    return bio[-6]

#%%
#Fonction donnant le poids d'un joueur
def poids(soup):
    info=soup.find('div', {'class': "col-12"})
    bio = info.find('div', {'class': 'meta ellipsis'})
    bio_str = bio.text.strip()
    bio = bio_str.split()
    return bio[-1]

#%%
#Fonction donnant la taille d'un joueur
def taille(soup):
    info=soup.find('div', {'class': "col-12"})
    bio = info.find('div', {'class': 'meta ellipsis'})
    bio_str = bio.text.strip()
    bio = bio_str.split()
    return bio[-2]

# %%
# Fonction donnant overall_rating d'un joueur
def overall_rating(soup):
    info=soup.find('div', {'class': "bp3-card"})
    info_stat=info.find('section', {'class': 'card spacing'})
    all=info_stat.findAll('div', {'class': 'block-quarter'})
    a=all[0].find('span')
    return a.text.strip()

#%% 
# Fonction donnant potential d'un joueur
def potential(soup):
    info=soup.find('div', {'class': "bp3-card"})
    info_stat=info.find('section', {'class': 'card spacing'})
    all=info_stat.findAll('div', {'class': 'block-quarter'})
    a=all[1].find('span')
    return a.text.strip()

#%% 
# Fonction donnant la valeur d'un joueur
def value(soup):
    info=soup.find('div', {'class': "bp3-card"})
    info_stat=info.find('section', {'class': 'card spacing'})
    all=info_stat.findAll('div', {'class': 'block-quarter'})
    a=all[2].find('div')
    b=a.text.strip()
    return b.split('V')[0]

#%% 
# Fonction donnant le salaire d'un joueur
def wage(soup):
    info=soup.find('div', {'class': "bp3-card"})
    info_stat=info.find('section', {'class': 'card spacing'})
    all=info_stat.findAll('div', {'class': 'block-quarter'})
    a=all[3].find('div')
    b=a.text.strip()
    return b.split('W')[0]

#%%
# Fonction donnant la valeur de la clause de rupture
def release_clause(soup):
    info=soup.find('ul', {'class': 'pl'})
    info_sup=info.findAll('li', {'class': 'ellipsis'})
    a=info_sup[-2].find('span')
    return a.text.strip()

#%%
# Fonction donnant la position de jeu d'un joueur
def position(soup):
    position=soup.findAll('div', {'class':'col col-4'})
    a=position[1].find('span',{'class':'pos'})
    return a.text.strip()

#%%
def attacking(soup):
    skills=soup.findAll('div', {'class':'col col-12'})
    attacking=skills[1].findAll('ul', {'class':'pl'})[0]
    all=attacking.findAll('span',{'class': 'bp3-tag'})
    L=[int(all[k].text.strip()) for k in range(len(all))]
    return np.mean(L)

# %%
def technique(soup):
    skills=soup.findAll('div', {'class':'col col-12'})
    attacking=skills[1].findAll('ul', {'class':'pl'})[1]
    all=attacking.findAll('span',{'class': 'bp3-tag'})
    L=[int(all[k].text.strip()) for k in range(len(all))]
    return np.mean(L)

#%%
def mouvement(soup):
    skills=soup.findAll('div', {'class':'col col-12'})
    attacking=skills[1].findAll('ul', {'class':'pl'})[2]
    all=attacking.findAll('span',{'class': 'bp3-tag'})
    L=[int(all[k].text.strip()) for k in range(len(all))]
    return np.mean(L)

#%%
def puissance(soup):
    skills=soup.findAll('div', {'class':'col col-12'})
    attacking=skills[1].findAll('ul', {'class':'pl'})[3]
    all=attacking.findAll('span',{'class': 'bp3-tag'})
    L=[int(all[k].text.strip()) for k in range(len(all))]
    return np.mean(L)

#%%
def etat_esprit(soup):
    skills=soup.findAll('div', {'class':'col col-12'})
    attacking=skills[1].findAll('ul', {'class':'pl'})[4]
    all=attacking.findAll('span',{'class': 'bp3-tag'})
    L=[int(all[k].text.strip()) for k in range(len(all))]
    return np.mean(L)

#%%
def defense(soup):
    skills=soup.findAll('div', {'class':'col col-12'})
    attacking=skills[1].findAll('ul', {'class':'pl'})[5]
    all=attacking.findAll('span',{'class': 'bp3-tag'})
    L=[int(all[k].text.strip()) for k in range(len(all))]
    return np.mean(L)

#%%
def gardien(soup):
    skills=soup.findAll('div', {'class':'col col-12'})
    attacking=skills[1].findAll('ul', {'class':'pl'})[6]
    all=attacking.findAll('span',{'class': 'bp3-tag'})
    L=[int(all[k].text.strip()) for k in range(len(all))]
    return np.mean(L)

#%%
def all_info(url):
    soup=soup_maker(url)
    return [name(soup),age(soup),poids(soup),taille(soup),overall_rating(soup),
    potential(soup),wage(soup),value(soup),release_clause(soup),position(soup),
    attacking(soup),technique(soup),mouvement(soup),puissance(soup),
    etat_esprit(soup),defense(soup),gardien(soup)]


#%%
df=pd.DataFrame({'url': all_ref(100)})
# %%
L=[[] for i in range(17)]

#%%
for url in df['url'].values:
    soup=soup_maker(url)
    L[0].append(name(soup))
    L[1].append(taille(soup))
    L[2].append(poids(soup))
    L[3].append(overall_rating(soup))
    L[4].append(potential(soup))
    L[5].append(wage(soup))
    L[6].append(value(soup))
    L[7].append(release_clause(soup))
    L[8].append(position(soup))
    L[9].append(attacking(soup))
    L[10].append(technique(soup))
    L[11].append(mouvement(soup))
    L[12].append(puissance(soup))
    L[13].append(defense(soup))
    L[14].append(gardien(soup))
    L[15].append(age(soup))
    L[16].append(etat_esprit(soup))

# %%
df['Nom']=L[0]
df['Taille']=L[1]
df['Age']=L[15]
df['Poids']=L[2]
df['Salaire']=L[5]
df['Valeur']=L[6]
df['Clause de rupture']=L[7]
df['Score total']=L[3]
df['Score pontentiel']=L[4]
df['Position']=L[8]
df['Attaque']=L[9]
df['Technique']=L[10]
df['Mouvement']=L[11]
df['Puissance']=L[12]
df['Defense']=L[13]
df['Gardien']=L[14]
df['Etat Esprit']=L[16]
#%%
# Exportation dans un excel
df.to_excel('Datasofifa.xlsx')

