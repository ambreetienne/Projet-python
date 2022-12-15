#%%
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import sklearn.metrics
from sklearn.linear_model import LinearRegression
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path
import seaborn as sns
import pandas as pd

# %%
df=pd.read_excel('Datasofifaclean.xlsx')
df=df.drop(columns=['Unnamed: 0'],axis=1)

#%%
# Normalisation des variables
col_modele=["Age","Taille","Salaire","Valeur","Score total","Attaque","Technique","Mouvement","Puissance","Defense","Gardien","Etat Esprit"]
df2=df[col_modele]
features = df[col_modele]
features = StandardScaler().fit(features.values).transform(features.values)
df2[col_modele] = features

#%%
#Echantillon de test et d'apprentissage
X_train, X_test, y_train, y_test = train_test_split(
    df2.drop(["Valeur"], axis = 1),
    100*df2[['Valeur']].values.ravel(), test_size=0.2, random_state=0
)

# %%
lasso1 = Lasso(fit_intercept=True,normalize=False, alpha = 0.1).fit(X_train,y_train)
# %%
features_selec = df2.select_dtypes(include=np.number).drop("Valeur", axis = 1).columns[np.abs(lasso1.coef_)>0].tolist()

# %%
corr = df2[features_selec].corr()

plt.figure()
p = corr.style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)
p
# %%
