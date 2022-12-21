#%%
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import lasso_path
from sklearn.linear_model import LassoCV



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

## REGRESSION LINEAIRE

#%%
# Première régression linéaire
model=LinearRegression()
model.fit(X_train,y_train)

#%%
# R² de la régression
model.score(X_train,y_train)
model.score(X_test,y_test)

#%%
#Valeur prédite en fonction de sa véritable valeur
predictions=model.predict(X_test)
x=range(int(min(predictions)),int(max(predictions)))

plt.plot(x,x,c='r')
plt.scatter(y_test,predictions)

#%%
# MSE de la régression
# Echantillon d'apprentissage
pred_train = model.predict(X_train)
mse_train = mean_squared_error(y_train, pred_train)

# Echantillon de test
pred = model.predict(X_test)
mse_test =mean_squared_error(y_test, pred)

#%%
#Sortie stata de la régression
X_train_cst = sm.add_constant(X_train)

model= sm.OLS(y_train,X_train_cst)
results= model.fit()
print(results.summary())

#%%
# Régression linéaire avec la variable Clause de rupture
#Normalisation
col_modele_cr=["Age","Taille","Clause de rupture","Salaire","Valeur","Score total","Attaque","Technique","Mouvement","Puissance","Defense","Gardien","Etat Esprit"]
df3=df[col_modele_cr]
features = df[col_modele_cr]
features = StandardScaler().fit(features.values).transform(features.values)
df3[col_modele_cr] = features

#%%
# Echantillon de test et d'apprentissage
X_train_cr, X_test_cr, y_train_cr, y_test_cr = train_test_split(
    df3.drop(["Valeur"], axis = 1),
    100*df3[['Valeur']].values.ravel(), test_size=0.2, random_state=0
)

#%%
#Sortie Stata de la régression
X_train_cr=sm.add_constant(X_train_cr)

model= sm.OLS(y_train_cr,X_train_cr)
results= model.fit()
print(results.summary())


## REGRESSION LASSO

# %%
#Lasso pour alpha=0.1
lasso1 = Lasso(fit_intercept=True,normalize=False, alpha = 0.1)

#Fit le modèle
lasso1.fit(X_train,y_train)

#%%
#R² de la régression
lasso1.score(X_test, y_test)
lasso1.score(X_train, y_train)

#%%
## MSE de la régression
# Echantillon d'apprentissage
pred_train = lasso1.predict(X_train)
mse_train = mean_squared_error(y_train, pred_train)

# Echantillon de test
pred = lasso1.predict(X_test)
mse_test =mean_squared_error(y_test, pred)
# %%
#Variables sélectionnées par la régression Lasso
features_selec = df2.select_dtypes(include=np.number).drop("Valeur", axis = 1).columns[np.abs(lasso1.coef_)>0].tolist()
# %%
#Corrélation des variables sélectionnées
corr = df2[features_selec].corr()

plt.figure()
p = corr.style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)
p

# %%
## Lasso avec différents alpha
my_alphas = np.array([0.001,0.01,0.02,0.025,0.05,0.1,0.25,0.5,0.8,1.0])
alpha_for_path, coefs_lasso, _ = lasso_path(X_train,y_train,alphas=my_alphas)
nb_non_zero = np.apply_along_axis(func1d=np.count_nonzero,arr=coefs_lasso,axis=0)
#%%
#Nombre de variables retenues en fonction de alpha
sns.set_style("whitegrid")
ax = plt.gca()

ax.plot(alpha_for_path, nb_non_zero)
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('Nb. variables retenues')
plt.title("Nombre de variables retenues fonction de alpha")
#%%
#MSE en fonction de alpha
mse_train=[]
mse_test=[]
lasso = Lasso(max_iter=10000)
for a in my_alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    pred = lasso.predict(X_test)
    mse_test.append(mean_squared_error(y_test, pred))
    pred_train = lasso.predict(X_train)
    mse_train.append(mean_squared_error(y_train, pred_train))

#%%
#MSE de l'échantillon d'apprentissage en fonction de alpha
sns.set_style("whitegrid")
ax = plt.gca()

ax.plot(alpha_for_path, mse_train)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('MSE')
plt.title("MSE de l'échantillon d'apprentissage fonction de alpha")

#%%
#MSE de l'échantillon de test en fonction de alpha
sns.set_style("whitegrid")
ax = plt.gca()

ax.plot(alpha_for_path, mse_test)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('MSE')
plt.title("MSE de l'échantillon de test fonction de alpha")

# %%
## Détermination du alpha optimal

# Lasso avec validation croisée en 5 blocs
model = LassoCV(cv=5, random_state=0, max_iter=10000)

# Fit model
model.fit(X_train, y_train)

#%%
#Meilleure valeur de pénalisation par validation croisée
model.alpha_

#%%
#Lasso avec le alpha optimal
lasso_best = Lasso(alpha=model.alpha_)
lasso_best.fit(X_train, y_train)

#%%
#Variables sélectionnées avec le alpha optimal
features_selec2 = df2.select_dtypes(include=np.number).drop("Valeur", axis = 1).columns[np.abs(lasso_best.coef_)>0].tolist()

#%%
#Evaluation du modèle avec le alpha optimal
#R² échantillon d'apprentissage
lasso_best.score(X_train, y_train)
#R² échantillon de test
lasso_best.score(X_test, y_test)

#%%
## MSE de la régression
# Echantillon d'apprentissage
pred_train = lasso_best.predict(X_train)
mse_train_best = mean_squared_error(y_train, pred_train)

# Echantillon de test
pred = lasso_best.predict(X_test)
mse_test_best =mean_squared_error(y_test, pred)

#%%
#MSE en fonction de alpha pour les 5 blocs de validation
plt.semilogx(model.alphas_, model.mse_path_, ":")
plt.plot(
    model.alphas_ ,
    model.mse_path_.mean(axis=-1),
    "k",
    label="Moyenne à travers les blocs",
    linewidth=2,
)

plt.axvline(
    model.alpha_, linestyle="--", color="k", label="alpha optimal"
)

plt.legend()
plt.xlabel("alpha")
plt.ylabel("MSE")
plt.title("MSE de chaque bloc de validation")
plt.axis("tight")