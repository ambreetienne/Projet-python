#%%
import urllib
import bs4
import pandas as pd
import numpy as np
from urllib import request
import requests

# %%
def convert_age(age):
    a=age.split('y')
    return int(a[0])

#%%
def convert_poids(poids):
    a=poids.split('k')
    return int(a[0])

#%%
def convert_taille(taille):
    a=taille.split('c')
    return int(a[0])

# %%
# Fonction permettant de transformer les str en int
def convert_value(wage):
    a=wage.split('€')
    if len(a[1].split('K'))==2:
        b=a[1].split('K')
        return float(b[0])*1000
    elif len(a[1].split('M'))==2:
        b=a[1].split('M')
        return float(b[0])*1000000
    else:
        return float(a[1])

# %%
df=pd.read_excel('Datasofifa.xlsx')
#%%
df.drop(columns=['Unnamed: 0'],axis=1)
#%%
df.dropna()
#%%
df=df.drop(df[df['Clause de rupture'] == 'Yes'].index)
df=df.drop(df[df['Clause de rupture'] == 'No'].index)
df=df.drop(df[df['Clause de rupture'] == 'N/A/ N/A'].index)

#%%
df.reset_index()
# %%
df['Valeur']=df['Valeur'].apply(convert_value)
df['Salaire']=df['Salaire'].apply(convert_value)
df['Clause de rupture']=df['Clause de rupture'].apply(convert_value)

# %%
df['Poste gardien']=np.where(df['Position']=='GK','1','0')
# %%
