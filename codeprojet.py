#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 12:18:07 2022

@author: lucasberne
"""

#importation des modukes et package

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import xlsxwriter
import csv
import seaborn as sns




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



model=LinearRegression()
model.fit(xTrain,yTrain)
print(model.score(xTrain,yTrain))




































