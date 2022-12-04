#%%
import urllib
import bs4
import pandas
from urllib import request
import requests

#%%
url = 'http://sofifa.com/players?offset=0'
# %%
r = requests.get(url)

# %%
markup = r.content

# %%
soup = bs4.BeautifulSoup(markup, 'lxml')


# %%
table = soup.find('table', {'class': 'table-hover'})
# %%
tbody = table.find('tbody')
# %%
all_a = tbody.find_all('a', {'class': ''})
# %%
[‘http://sofifa.com' + player[‘href’] for player in all_a]
# %%
