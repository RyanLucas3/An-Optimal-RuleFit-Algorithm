#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from ORRFA import *
import interpretableai
from interpretableai import iai
import pandas as pd
import json
from HR import *


# In[15]:


dataset.index


# In[26]:


dataset = pd.read_csv("climate_change.csv")

train_idx = np.random.choice(a = np.arange(0, len(dataset)), size = int(0.6*len(dataset)))
test_idx = [i for i in np.arange(0, len(dataset)) if i not in train_idx]

features = dataset.loc[train_idx].iloc[:, :-2]
diagnosis = dataset.loc[train_idx].iloc[:, -1]

orig_features_size = features.copy().shape[1]


# In[28]:


print(len(train_idx))
print(len(test_idx))


# In[23]:


features_test = dataset.loc[test_idx].iloc[:, :-2]
diagnosis_test = dataset.loc[test_idx].iloc[:, -1]


# In[24]:


print(len(dataset))
print(len(features))
print(len(features_test))


# In[5]:


features


# In[ ]:


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


# In[ ]:


np.random.seed(1)
n = 5
rules = {}
for i in range(1,15):
    #################
    dataset = dataset.sample(n=int(n),axis='columns')
    OCT_H = iai.GridSearch(
          iai.OptimalTreeRegressor(
              max_depth = 3, cp = 0.00000000001))

    OCT_H.fit(features, diagnosis)
    OCT_H.write_json(f"tre_{i}.json")

    ######################
    f = open(f"tre_{i}.json")
    data = json.load(f)

    paths = gen_paths(data)

    rule_features_new = gen_features(
            paths, data, features, orig_features_size)

    rules[f"tre_{i}"] = rule_features_new

    print(f"Tree: {i}")


# In[ ]:


features.columns


# In[ ]:


big_df = pd.DataFrame()
for tree in rules.keys():

    big_df = pd.concat([big_df, rules[tree]],axis =1)


# In[ ]:


df = big_df.loc[:,~big_df.columns.duplicated()].copy()


# In[ ]:


df


# In[ ]:


for column in df.columns[1:]:
    if df[column].sum() == len(df):
        del df[column]


# In[ ]:


df


# In[ ]:


features_for_robust_reg = df


# In[ ]:


rule_features = features_for_robust_reg.copy()
features_for_robust_reg = pd.concat([features, features_for_robust_reg], axis = 1)


# In[ ]:


model = HolisticRobust(
                       α = 0,
                       ϵ = 0,
                       r = 0.05,
                       classifier = "Linear",
                       learning_approach = "HR") # Could be ERM either


# In[ ]:


features.insert(loc = 0, column = 'Intecept', value = [1 for i in range(len(features_for_robust_reg))])


# In[ ]:


θ, obj_value = model.fit(ξ = (np.matrix(features_for_robust_reg), np.array(diagnosis)))


# In[ ]:


θ


# In[ ]:


predictions = np.multiply(θ, features_for_robust_reg
                          ).sum(axis = 1)


# In[ ]:


plt.plot(diagnosis - predictions)


# In[ ]:


(diagnosis - predictions).describe()


# In[ ]:


SSE_model = sum((diagnosis - predictions)**2)
SSE_mean = sum((diagnosis - diagnosis.mean())**2)

1 - SSE_model/SSE_mean


# In[ ]:


features['CFC_12'].describe()


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


# In[ ]:


θ

