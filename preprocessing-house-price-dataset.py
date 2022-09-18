#!/usr/bin/env python
# coding: utf-8

# # In this notebook, <br>
# I will show my way to preprocess the House-Price dataset. <br>
# I'm very beginner to this. <br>
# **Please feel free to discuss or leave an advice in the comment section :)** <br>
# Thanks in advance

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


get_ipython().system('pip install seaborn')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


# In[4]:


data = pd.read_csv('/Users/ryanlucas/Desktop/training.csv')


# In[5]:


data


# # Declare independent variables(X) and target(y)

# In[6]:


from sklearn.model_selection import train_test_split
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']
X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.2)


# # Separate numerical features(num) and categorical features(cat)

# In[7]:


num = X_train.select_dtypes(exclude='object')
cat = X_train.select_dtypes(include='object')


# # Numerical features

# ## 0) Drop 'Id' column

# In[8]:


num.drop('Id', axis=1, inplace=True)


# ## 1) Missing values

# In[9]:


from sklearn.impute import SimpleImputer

Imputer = SimpleImputer(missing_values=np.nan, strategy='mean').fit(num)

num_imputed = Imputer.transform(num)

print(np.isnan(num_imputed).sum()/len(num))


# The result from SimpleImputer is Numpy array. Let's bring it back to interpletable Pandas DataFrame

# In[10]:


num_imputed = pd.DataFrame(num_imputed, columns=num.columns)
num_imputed.head(3)


# ## 2) Discard the very low variance

# Quickly visualize the variance of numerical features

# In[11]:


cm = sns.light_palette("green", as_cmap=True)

num_var = pd.DataFrame((num/num.mean()).var(), columns=['norm_variance']).apply(lambda x:np.round(x,4))
num_var = num_var.style.background_gradient(cmap=cm)
num_var


# We'll drop the relatively low variance columns (<0.05)

# In[12]:


from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=0.05).fit(num/num.mean())

sel_col = sel.get_support()

num_imputed_highVar = num_imputed.loc[:,sel_col]


# ## 3) Standard scale

# We then scale the numerical features.

# In[13]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(num_imputed_highVar)

num_imputed_highVar_scaled = scaler.transform(num_imputed_highVar)
num_imputed_highVar_scaled = pd.DataFrame(num_imputed_highVar_scaled, columns=num_imputed_highVar.columns)


# ## 4) Inspect correlation

# In[14]:


correlation = np.abs(num_imputed_highVar_scaled.corr())
correlation.style.background_gradient(cmap=cm)


# We see that few features are correlated to each other. So, we have to drop them.

# In[15]:


highly_correlated_name = dict()
highly_correlated_index = list()
visited = set()


for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        if correlation.columns[i] != correlation.columns[j] and correlation.columns[i] not in visited and np.abs(correlation.loc[correlation.columns[i],correlation.columns[j]]) >=0.85:
            highly_correlated_name[correlation.columns[i]]=correlation.columns[j]
            highly_correlated_index.append(i)
            visited.update([correlation.columns[i],correlation.columns[j]])

num_imputed_highVar_scaled_noCorr = num_imputed_highVar_scaled.drop(highly_correlated_name.keys(), axis=1)
numeric_preprocessed = num_imputed_highVar_scaled_noCorr.copy()


# Now we have done
# - Drop 'Id'
# - Imputer  ->  **Imputer Obj.**
# - Filter out very low variance  ->  **sel_col Mask**
# - Standard scale  ->  **scaler Obj.**
# - Drop highly correlated columns  ->  **highly_correlated Dict**

# # Categorical features

# ## 1) Missing values

# In[16]:


missing = pd.DataFrame(cat.isnull().sum()/len(cat), columns=['NaN'])
missing.style.background_gradient(cmap=cm)


# We saw that few features are missing too much. So, we'll drop them.

# In[17]:


to_drop_cat = missing[missing['NaN']>=0.47]
cat_dropTooMiss = cat.drop(to_drop_cat.index, axis=1)


# ## 2) Fill 'No'
# After dropping those columns, remaining columns still have few missing value. <br>
# We'll create a new class for them ('No')

# In[18]:


cat_dropTooMiss_replaceNo = cat_dropTooMiss.fillna('No')


# Then we need to add 'No' to reference dataset for OneHotEncoder to have OneHotEncoder familiar with new added class.

# In[19]:


No = pd.DataFrame(['No' for i in range(len(cat_dropTooMiss_replaceNo.columns))],cat_dropTooMiss_replaceNo.columns)
cat_dropTooMiss_replaceNo_forOHE = cat_dropTooMiss_replaceNo.append(No.T, ignore_index=True)

cat_dropTooMiss_replaceNo_forOHE.tail(3)


# ## 3) One Hot encoding

# In[20]:


from sklearn.preprocessing import OneHotEncoder

cat_all = data.select_dtypes(include='object')
cat_all.drop(to_drop_cat.index, axis=1, inplace=True)
cat_all.fillna('No', inplace=True)
cat_all = cat_all.append(No.T, ignore_index=True)

ohe = OneHotEncoder(sparse=False).fit(cat_all)
ohe.categories_


# In[21]:


cat_dropTooMiss_replaceNo_ohe = ohe.transform(cat_dropTooMiss_replaceNo)
categorical_preprocessed = cat_dropTooMiss_replaceNo_ohe.copy()


# Now we have done
# - Drop to many missing value  ->  **to_drop_cat**
# - Fill missing value with 'No' 
# - One how encoding  ->  **ohe Obj.**

# # Categorical + Numerical

# In[22]:


X_train_preprocessed = np.concatenate([categorical_preprocessed, numeric_preprocessed.values], axis=1)


# Scale the target feature (y)

# In[23]:


y_scaler = StandardScaler().fit(y_train.values.reshape(-1,1))
y_train_scaled = y_scaler.transform(y_train.values.reshape(-1,1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1,1))


# Create PREPROCESS() function that summarize what we've done so far.

# In[24]:


def PREPROCESS(DATA: pd.DataFrame):
    num = DATA.select_dtypes(exclude='object')
    cat = DATA.select_dtypes(include='object')
    
    # NUMERIC
        # 0) Drop 'Id'
    num.drop('Id', axis=1, inplace=True)
        # 1) Impute
    num_imputed = Imputer.transform(num)
        # 2) Filter out very low variance
    num_imputed_highVar = num_imputed[:,sel_col]
        # 3) Standard scale
    num_imputed_highVar_scaled = scaler.transform(num_imputed_highVar)
        # 4) Drop highly correlated columns
    num_imputed_highVar_scaled_noCorr = np.delete(num_imputed_highVar_scaled, highly_correlated_index, axis=1)
    # Preprocessed
    numeric_preprocessed = num_imputed_highVar_scaled_noCorr.copy()
    
    
    # CATEGORICAL
        # 0) Drop too many missing
    cat_dropTooMiss = cat.drop(to_drop_cat.index, axis=1)
        # 1) Fill remaining missing value with 'No'
    cat_dropTooMiss_replaceNo = cat_dropTooMiss.fillna('No')
        # 2) One hot encoding
    cat_dropTooMiss_replaceNo_ohe = ohe.transform(cat_dropTooMiss_replaceNo)
    # Preprocessed
    categorical_preprocessed = cat_dropTooMiss_replaceNo_ohe.copy()
    
    return np.concatenate([categorical_preprocessed, numeric_preprocessed], axis=1)


# Finally, let's preprocess the test set as well.

# In[25]:


X_test_preprocessed = PREPROCESS(X_test)


# In[26]:


X_test_preprocessed


# In[26]:




