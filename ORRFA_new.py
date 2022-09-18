#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from ORRFA import *
import interpretableai
from interpretableai import iai
import pandas as pd
import json
from HR import *
import multiprocessing


# In[2]:


dataset = pd.read_csv("kc_house_data.csv")


# In[3]:


dataset


# In[4]:


train_idx = np.arange(0, int(0.7*len(dataset)))
test_idx = np.arange(int(0.7*len(dataset)), int(1*len(dataset)))

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
orig_columns = features.copy().columns
orig_features_size = features.copy().shape[1]


# In[5]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features = pd.DataFrame(sc.fit_transform(features))
features.columns = orig_columns


# In[6]:


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


# In[7]:


features_test = pd.DataFrame(sc.fit_transform(features_test))
features_test.columns = orig_columns


# In[8]:


diagnosis


# In[9]:


print(len(dataset))
print(len(features))
print(len(features_test))


# In[10]:


features


# In[ ]:


np.random.seed(1)
n = 5
rules = {}


def fit_trees(i):

    for i in range(1,50):

        #################
        sampled_features = features.copy().sample(n=int(n),axis='columns')

        OCT_H = iai.GridSearch(
            iai.OptimalTreeRegressor(
                max_depth = 4, cp = 0.00000000001))

        OCT_H.fit(sampled_features, diagnosis)
        OCT_H.write_json(f"tre_{i}.json")

        ######################
        f = open(f"tre_{i}.json")
        data = json.load(f)

        paths = gen_paths(data)

        sub_paths = gen_subpaths(paths)

        names = sampled_features.columns

        rule_features_new = gen_features(
                sub_paths, data, sampled_features, names)


        rules[f"tre_{i}"] = rule_features_new


with multiprocessing.Pool() as pool:
	# call the function for each item in parallel
	pool.map(fit_trees, items)



# In[ ]:


features.columns


# In[ ]:


big_df = pd.DataFrame()
for tree in rules.keys():

    big_df = pd.concat([big_df, rules[tree]],axis =1)


# In[ ]:


df = big_df.loc[:,~big_df.columns.duplicated()].copy()


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


features_for_robust_reg.sum()


# In[ ]:


features_for_robust_reg.insert(loc = 0, column = 'Intercept', value = [1 for i in range(len(features_for_robust_reg))])


# In[ ]:


features_for_robust_reg


# In[ ]:


features_for_robust_reg


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


features.insert(loc = 0, column = 'Intercept', value = [1 for i in range(len(features))])


# In[ ]:


reg_baseline = LinearRegression().fit(features.copy()
, diagnosis)


# In[ ]:


θ_baseline = reg_baseline.coef_


# In[ ]:


predictions_baseline = reg_baseline.predict(features.copy())


# In[ ]:


(diagnosis - predictions_baseline).describe()


# In[ ]:


SSE_model = sum((diagnosis - predictions_baseline)**2)
SSE_mean = sum((diagnosis - diagnosis.mean())**2)

1 - SSE_model/SSE_mean


# In[ ]:


features_test.insert(loc = 0, column = 'Intercept', value = [1 for i in range(len(features_test))])


# In[ ]:


predictions_test_baseline = reg_baseline.predict(features_test)


# In[ ]:


SSE_model = sum((diagnosis_test - predictions_test_baseline)**2)
SSE_mean = sum((diagnosis_test - diagnosis.mean())**2)

1 - SSE_model/SSE_mean


# In[ ]:


reg_trees = LinearRegression().fit(features.copy()
, diagnosis)


# In[ ]:


from sklearn import linear_model


# In[ ]:


lasso_trees = linear_model.Ridge(alpha = 1).fit(features_for_robust_reg.copy()
, diagnosis)


# In[ ]:


predictions_lasso_trees = lasso_trees.predict(features_for_robust_reg)


# In[ ]:


SSE_model = sum((predictions_lasso_trees - diagnosis)**2)
SSE_mean = sum((diagnosis - diagnosis.mean())**2)

1 - SSE_model/SSE_mean


# In[ ]:


features = features_test
for rule in rule_features.columns:

    index_feature = list(features.loc[eval(rule)].index)

    print(index_feature)
    new_feature = [1 if i in index_feature else 0 for i in range(features.index[0], features.index[-1]+1)]

    features[rule] = new_feature



# In[ ]:


new_features = features_test


# In[ ]:


new_features.insert(loc = 0, column = 'Intercept', value = [1 for i in range(len(new_features))])


# In[ ]:


predictions_test = lasso_trees.predict(features_test)


# In[ ]:


plt.scatter(predictions_test, diagnosis_test)


# In[ ]:


SSE_model = sum((diagnosis_test - predictions_test)**2)
SSE_mean = sum((diagnosis_test - diagnosis.mean())**2)

1 - SSE_model/SSE_mean


# In[ ]:




