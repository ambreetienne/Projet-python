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
from sklearn.cluster import KMeans 

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
#6. Utilisation de lasso_path
my_alphas = np.array([0.001,0.01,0.02,0.025,0.05,0.1,0.25,0.5,0.8,1.0])
alpha_for_path, coefs_lasso, _ = lasso_path(X_train,y_train,alphas=my_alphas)
print(coefs_lasso)
nb_non_zero = np.apply_along_axis(func1d=np.count_nonzero,arr=coefs_lasso,axis=0)
print(nb_non_zero)

#%%
sns.set_style("whitegrid")
plt.figure()
p = sns.lineplot(y=nb_non_zero, x=alpha_for_path)
p.set(title = r"Number variables and regularization parameter ($\alpha$)", xlabel=r'$\alpha$', ylabel='Nb. de variables')

# %%
from sklearn.linear_model import LassoCV

df3 = df2.select_dtypes(include=np.number)
df3.replace([np.inf, -np.inf], np.nan, inplace=True)
df3 = df3.fillna(0)
scaler = StandardScaler()
yindex = df3.columns.get_loc("Valeur")
df3_scale = scaler.fit(df3).transform(df3)
# X_train, X_test , y_train, y_test = train_test_split(np.delete(data, yindex, axis = 1),data[:,yindex], test_size=0.2, random_state=0)

lcv = LassoCV(alphas=my_alphas ,normalize=False,fit_intercept=False,random_state=0,cv=5).fit(np.delete(df3_scale, yindex, axis = 1), df3_scale[:,yindex])

# %%
print("alpha optimal :", lcv.alpha_)

#%%
lasso2 = Lasso(fit_intercept=True, alpha = lcv.alpha_).fit(X_train,y_train)
features_selec2 = df2.select_dtypes(include=np.number).drop("Valeur", axis = 1).columns[np.abs(lasso2.coef_)>0].tolist()


#%%
### Test clustering
col_name=["Age","Taille","Salaire","Score total","Attaque","Technique","Mouvement","Puissance","Defense","Gardien","Etat Esprit"]
col_modele=["Age","Taille","Salaire","Valeur","Score total","Attaque","Technique","Mouvement","Puissance","Defense","Gardien","Etat Esprit"]
df4=df[col_modele]

#%%
model = KMeans(n_clusters=5)
model.fit(df4[col_name])
# %%
df4['label'] = model.labels_

# %%
plt.figure()
p = sns.scatterplot(
  data=df4,
  x="Salaire",
  y="Attaque", hue = "label", palette="deep",
  alpha = 0.4)

# %%
plt.figure()
p2 = sns.displot(data=df4, x="Valeur", hue="label", alpha = 0.4)
# %%
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

#%%
visualizer = KElbowVisualizer(model, k=(2,8))
visualizer.fit(df4[col_name]) 
# %%
