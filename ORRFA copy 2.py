#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from ORRFA import *
import interpretableai
from interpretableai import iai
import pandas as pd
import json
from HR import *


# In[16]:


dataset = pd.read_csv("/Users/ryanlucas/Desktop/kc_house_data.csv")


# In[64]:


dataset


# In[18]:


train_idx = np.arange(0, int(0.2*len(dataset)))
test_idx = np.arange(int(0.2*len(dataset)), int(0.3*len(dataset)))

features = dataset.loc[train_idx, ["floors", "bedrooms",
    "bathrooms",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "sqft_living",
    "sqft_lot",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15"]]

diagnosis = dataset.loc[train_idx, "price"]

orig_features_size = features.copy().shape[1]


# In[19]:


features_test = dataset.loc[test_idx, ["floors", "bedrooms",
    "bathrooms",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "sqft_living",
    "sqft_lot",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15"]]
diagnosis_test = dataset.loc[test_idx, 'price']


# In[20]:


diagnosis


# In[21]:


print(len(dataset))
print(len(features))
print(len(features_test))


# In[22]:


features


# In[23]:


def gen_features(sub_paths, data, features, orig_feature_size):
    new_features = features.copy()
    for path in sub_paths:
        statement = gen_statement(path, data, features)
        if statement != '(0 < 0.0)' and statement != '(0 >= 0.0)':

            try:
                index_feature = features.loc[eval(statement)].index


                new_feature = [1 if i in index_feature else 0 for i in range(len(features))]

                if sum(new_feature) >= 1:
                    new_features[statement] = new_feature

            except:
                print(statement)
    return new_features.iloc[:, orig_feature_size:]


# In[24]:


np.random.seed(1)
n = 5
rules = {}
for i in range(1,15):

    #################
    sampled_features = features.copy().sample(n=int(n),axis='columns')

    OCT_H = iai.GridSearch(
          iai.OptimalTreeRegressor(
              max_depth = 3, cp = 0.00000000001))

    OCT_H.fit(sampled_features, diagnosis)
    OCT_H.write_json(f"tre_{i}.json")

    ######################
    f = open(f"tre_{i}.json")
    data = json.load(f)

    paths = gen_paths(data)

    rule_features_new = gen_features(
            paths, data, features, orig_features_size)

    rules[f"tre_{i}"] = rule_features_new

    print(f"Tree: {i}")


# In[66]:


OCT_H.get_learner()


# In[52]:


features.columns


# In[63]:


data['lnr']['tree_']['nodes'][11]


# In[25]:


features.columns


# In[26]:


big_df = pd.DataFrame()
for tree in rules.keys():

    big_df = pd.concat([big_df, rules[tree]],axis =1)


# In[27]:


df = big_df.loc[:,~big_df.columns.duplicated()].copy()


# In[28]:


df


# In[29]:


for column in df.columns[1:]:
    if df[column].sum() == len(df):
        del df[column]


# In[30]:


df


# In[31]:


features_for_robust_reg = df


# In[32]:


rule_features = features_for_robust_reg.copy()
features_for_robust_reg = pd.concat([features, features_for_robust_reg], axis = 1)


# In[33]:


features_for_robust_reg.sum()


# In[34]:


features_for_robust_reg.insert(loc = 0, column = 'Intercept', value = [1 for i in range(len(features_for_robust_reg))])


# In[35]:


features_for_robust_reg


# In[42]:


features_for_robust_reg


# In[43]:


model = HolisticRobust(
                       α = 0,
                       ϵ = 0,
                       r = 0.05,
                       classifier = "Linear",
                       learning_approach = "HR") # Could be ERM either


# In[44]:


θ, obj_value = model.fit(ξ = (np.matrix(features_for_robust_reg), np.array(diagnosis)))


# In[38]:


predictions = np.multiply(θ, features_for_robust_reg
                          ).sum(axis = 1)


# In[ ]:


diagnosis.plot()


# In[ ]:


(diagnosis - predictions).describe()


# In[ ]:


SSE_model = sum((diagnosis - predictions)**2)
SSE_mean = sum((diagnosis - diagnosis.mean())**2)

1 - SSE_model/SSE_mean


# In[ ]:


features


# In[ ]:


rule_features


# In[ ]:


features = features_test

for rule in rule_features.columns:

    index_feature = features.loc[eval(rule)].index

    new_feature = [1 if i in index_feature else 0 for i in range(len(features))]

    if sum(new_feature) >= 1:
        features[rule] = new_feature


# In[ ]:


features

