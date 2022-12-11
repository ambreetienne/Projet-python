#%%
import urllib
import bs4
import pandas as pd
import numpy as np
from urllib import request
import requests

#%%
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