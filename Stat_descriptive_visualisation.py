#%%

#importation des modules et package



import pandas as pd 

import numpy as np

from matplotlib import pyplot as plt

import xlsxwriter

import csv

import seaborn as sns

from sklearn import preprocessing

import matplotlib.patches as  mpatches





#%%

#Récupération de la dataframe

df=pd.read_excel('Datasofifaclean.xlsx')

df=df.drop(columns=['Unnamed: 0'],axis=1)



## PARTIE STATISTIQUES DESCRIPTIVES ET VISUALISATION DES DONNEES

#%%

df.describe()



#%%

#Standardisation de la variable salaire

df['salaire_standard'] = preprocessing.scale(df['Salaire'])

f, axes = plt.subplots(2, figsize=(10, 10))

sns.distplot(df["Salaire"] , color="skyblue", ax=axes[0])

sns.distplot(df["salaire_standard"] , color="olive", ax=axes[1])



#%% 

# Standardisation de la variable age

df['age_standard'] = preprocessing.scale(df['Age'])

f, axes = plt.subplots(2, figsize=(10, 10))

sns.distplot(df["Age"] , color="skyblue", ax=axes[0])

sns.distplot(df["age_standard"] , color="olive", ax=axes[1])



# %%

# Standardisation de la valeur marchande des joueurs

df['valeur_standard'] = preprocessing.scale(df['Valeur'])

f, axes = plt.subplots(2, figsize=(10, 10))

sns.distplot(df["Valeur"] , color="skyblue", ax=axes[0])

sns.distplot(df["valeur_standard"] , color="olive", ax=axes[1])





# %%

# Standardisation du score total

df['score_standard'] = preprocessing.scale(df['Score total'])

f, axes = plt.subplots(2, figsize=(10, 10))

sns.distplot(df["Score total"] , color="skyblue", ax=axes[0])

sns.distplot(df["score_standard"] , color="olive", ax=axes[1])



#%%
