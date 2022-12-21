#%%
import pandas as pd
import numpy as np

# %%
# Fonction convertissant l'age en int
def convert_age(age):
    a=age.split('y')
    return int(a[0])

#%%
#Fonction convertissant le poids en int
def convert_poids(poids):
    a=poids.split('k')
    return int(a[0])

#%%
# Fonction convertissant la taille en int
def convert_taille(taille):
    a=taille.split('c')
    return int(a[0])

# %%
# Fonction permettant de transformer les valeurs en int
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
# Importation de la dataframe
df=pd.read_excel('Datasofifa.xlsx')
#%%
df=df.drop(columns=['Unnamed: 0'],axis=1)
#%%
df=df.dropna()
#%%
# On enlève les joueurs qui n'ont pas de clause de rupture
df=df.drop(df[df['Clause de rupture'] == 'Yes'].index)
df=df.drop(df[df['Clause de rupture'] == 'No'].index)
df=df.drop(df[df['Clause de rupture'] == 'N/A/ N/A'].index)

#%%
df=df.reset_index()
# %%
# On convertit les valeurs en int grâce à la fonction prévue à cet effet
df['Valeur']=df['Valeur'].apply(convert_value)
df['Salaire']=df['Salaire'].apply(convert_value)
df['Clause de rupture']=df['Clause de rupture'].apply(convert_value)

#%%
# On convertit l'âge, la taille et le poids en int grâce à la fonction prévue à cet effet
df['Age']=df['Age'].apply(convert_age)
df['Taille']=df['Taille'].apply(convert_taille)
df['Poids']=df['Poids'].apply(convert_poids)
# %%
# On crée une nouvelle variable Poste gardien qui vaut 1 si le joueur est un gardien
df['Poste gardien']=np.where(df['Position']=='GK','1','0')

##%%
# On exporte la base de données maintenant nettoyée au format excel
df.to_excel('Datasofifaclean.xlsx')