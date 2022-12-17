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

# Graphique de la densité du score total et du salaire

fig = sns.kdeplot(df['salaire_standard'], shade=True, color="r")

fig = sns.kdeplot(df['score_standard'], shade=True, color="b")

handles = [mpatches.Patch(facecolor=plt.cm.Reds(100), label="Salaire"),

           mpatches.Patch(facecolor=plt.cm.Blues(100), label="Score total")]

plt.legend(handles=handles)

plt.show()



# Score total sont + étalés que les salaires

#%%

# Corrélation des variables

variable_corr=['Valeur','Taille','Age','Salaire','Clause de rupture',

'Score total','Score pontentiel','Attaque','Technique','Puissance','Defense']

df_corr=df[variable_corr]

#%%

matrice_corr = df_corr.corr()
plt.figure(figsize=(15,5))

heatmap = sns.heatmap(matrice_corr, vmin=-1, vmax=1, annot=True)

#%%

axes=pd.plotting.scatter_matrix(df_corr, figsize=(9,9))
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')

plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.show()


#%%

# Subplot valeur en fonction des autres variables

plt.figure(figsize=(10,10))

for i in range(1,11):
    
    plt.subplot(3,4,i)

    plt.scatter(df[variable_corr[i]],df['Valeur'],s=0.5)

    plt.xlabel(variable_corr[i])

    plt.ylabel("Valeur")
# %%
