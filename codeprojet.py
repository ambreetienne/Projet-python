#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 12:18:07 2022

@author: lucasberne
"""
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
heatmap = sns.heatmap(matrice_corr, vmin=-1, vmax=1, annot=True)

#%%
pd.plotting.scatter_matrix(df_corr)

#%%
# Subplot valeur en fonction des autres variables
for i in range(1,11):
    plt.subplot(3,4,i)
    plt.scatter(df[variable_corr[i]],df['Valeur'],s=0.5)
    plt.xlabel(variable_corr[i])
    plt.ylabel('Valeur')



#%%
#Récupération des données depuis un fichier csv

data_fifa20= pd.read_csv("fifa20_data.csv")

colonne=["Name","Position","Age","Overall","Potential","ID","Height","Weight","Value","Wage","Release Clause","PAC","SHO","PAS","DRI","DEF","PHY"]

data1=data_fifa20[colonne]

data=data1.iloc[0:200,0:17]

#donner que avec des colonnes numérique

data_numérique=data1[["Age","Overall","PAC","SHO","PAS","DRI","DEF","PHY"]]

Z=data_numérique.values


plt.scatter(data["Age"],data["Value"])


###PARTIE 1 corrélation entre les variables


#Pour créer la matrice de corrélation entre les variables

Y=data1.corr()

# sns.heatmap(data.corr())
# plt.figure(figsize=(50, 50))
heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)

pd.plotting.scatter_matrix(data)



###PARTIE2 évaluer la qualité d'un modèle

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(data_numérique.drop("Overall",axis=1).values, data_numérique["Overall"].values, test_size = 0.2,random_state = 0)



##### PARTIE REGRESSION LINEAIRE


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import preprocessing



model=LinearRegression()
model.fit(xTrain,yTrain)
print(model.score(xTrain,yTrain))




































