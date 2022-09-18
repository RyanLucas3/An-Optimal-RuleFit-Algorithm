#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


# In[2]:


import numpy as np
# interpretableai.install_julia()
# interpretableai.install_system_image()


# In[3]:


import interpretableai
from interpretableai import iai


# In[4]:


import pandas as pd


# In[38]:


data = np.random.randint(low = 0, high = 100, size = (1000, 10))
data = pd.DataFrame(data)
features = data.iloc[:, 0:8].values
x = data.iloc[:, 8].copy()
diagnosis = x.values


# In[41]:


OCT_H = iai.GridSearch(
          iai.OptimalTreeClassifier(
              max_depth = 2))
OCT_H.fit_cv(features, diagnosis)


# In[42]:


OCT_H.write_json('learner.json')


# In[44]:


import json
f = open('learner.json')
data = json.load(f)


# In[50]:


def take_json_find_splits(json_file):

    nodes = data['lnr']['tree_']['nodes']

    for node in nodes:
        feature = data['lnr']['tree_']['nodes'][node]['split_mixed']['parallel_split']['feature']
        threshold = data['lnr']['tree_']['nodes'][node]['split_mixed']['parallel_split']['threshold']

    f"{feature}"


# In[54]:





# In[ ]:




