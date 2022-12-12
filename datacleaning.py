#%%
import urllib
import bs4
import pandas as pd
import numpy as np
from urllib import request
import requests
import matplotlib.pyplot as plt
import seaborn as sns

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
df=df.drop(columns=['Unnamed: 0'],axis=1)
#%%
df=df.dropna()
#%%
df=df.drop(df[df['Clause de rupture'] == 'Yes'].index)
df=df.drop(df[df['Clause de rupture'] == 'No'].index)
df=df.drop(df[df['Clause de rupture'] == 'N/A/ N/A'].index)

#%%
df=df.reset_index()
# %%
df['Valeur']=df['Valeur'].apply(convert_value)
df['Salaire']=df['Salaire'].apply(convert_value)
df['Clause de rupture']=df['Clause de rupture'].apply(convert_value)

#%%
df['Age']=df['Age'].apply(convert_age)
df['Taille']=df['Taille'].apply(convert_taille)
df['Poids']=df['Poids'].apply(convert_poids)
# %%
df['Poste gardien']=np.where(df['Position']=='GK','1','0')

##%%
df.to_excel('Datasofifa.xlsx')

# %%
from sklearn import preprocessing

#%%
df['age_standard'] = preprocessing.scale(df['Age'])
f, axes = plt.subplots(2, figsize=(10, 10))
sns.distplot(df["Age"] , color="skyblue", ax=axes[0])
sns.distplot(df["age_standard"] , color="olive", ax=axes[1])
# %%
df['poids_standard'] = preprocessing.scale(df['Poids'])
f, axes = plt.subplots(2, figsize=(10, 10))
sns.distplot(df["Poids"] , color="skyblue", ax=axes[0])
sns.distplot(df["poids_standard"] , color="olive", ax=axes[1])