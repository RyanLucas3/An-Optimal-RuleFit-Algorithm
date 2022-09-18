#!/usr/bin/env python
# coding: utf-8

# In[484]:


import numpy as np


# In[485]:


import interpretableai
from interpretableai import iai


# In[486]:


import pandas as pd


# In[487]:


data = np.random.randint(low = 0, high = 100, size = (1000, 10))
data = pd.DataFrame(data)
features = data.iloc[:, 0:8].values
features = pd.DataFrame(features)
y = (0.4*data.iloc[:, 8]*0.3*data.iloc[:, 5]).copy()
diagnosis = y.values


# In[488]:


features


# In[489]:


features.columns = [f'x_{i}' for i in range(len(features.columns))]


# In[490]:


features


# In[491]:


OCT_H = iai.OptimalTreeClassifier(max_depth = 3, cp = 0.00003)
OCT_H.fit(features, diagnosis)


# In[492]:


OCT_H.write_json('learner.json')


# In[493]:


import json
f = open('learner.json')
data = json.load(f)


# In[494]:


def gen_statement(path):

    nodes_on_path = path.split("+")
    statement = ""



    for node_num, node in enumerate(nodes_on_path):

        if node_num > 0:

            prev_node = nodes_on_path[node_num - 1]

            feature = data['tree_']['nodes'][int(node)-1]['split_mixed']['parallel_split']['feature']
            threshold = data['tree_']['nodes'][int(node)-1]['split_mixed']['parallel_split']['threshold']

            if int(node) == int(prev_node) + 1:
                equality = "<"

            else:
                equality = ">="

            if feature != 0:
                statement += f"(features['{features.columns[feature-1]}'] {equality} {threshold})"

            else:
                statement += f"({0} {equality} {threshold})"

            print(int(node))
            if int(node) != max([int(i) for i in nodes_on_path]):
                statement += " & "

    return statement


# In[498]:


paths = []

for child in [1]:

    node_1 = data['tree_']['nodes'][child-1]
    node_id1 = node_1['id']

    children = sorted([node_1['upper_child'], node_1['lower_child']])

    if children[0] == -2:
        id = f"{node_id1}"
        paths.append(id)
        break

    for child in children:
        node_2 = data['tree_']['nodes'][child-1]
        node_id2 = node_2['id']
        children = sorted([node_2['upper_child'], node_2['lower_child']])

        if children[0] == -2:
            id = f"{node_id1}+" + f"{node_id2}"
            paths.append(id)

        for child in children:

            node_3 = data['tree_']['nodes'][child-1]
            node_id3 = node_3['id']
            children = sorted([node_3['upper_child'], node_3['lower_child']])

            if children[0] == -2:
                id = f"{node_id1}+" + f"{node_id2}+" + f"{node_id3}"
                paths.append(id)

            for child in children:

                node_4 = data['tree_']['nodes'][child-1]
                node_id4 = node_4['id']
                children = sorted([node_4['upper_child'], node_4['lower_child']])

                if children[0] == -2:
                    id = f"{node_id1}+" + f"{node_id2}+" + f"{node_id3}+" + f"{node_id4}"
                    paths.append(id)



# In[499]:


for path in paths:
    statement = gen_statement(path)
    index_feature = features.loc[eval(statement)].index
    new_feature = [1  if i in index_feature else 0 for i in range(len(features))]
    features[statement] = new_feature


# In[506]:


from HR import *


# In[507]:


model = HolisticRobust(
                       α = 0.05,
                       ϵ = 0,
                       r = 0.1,
                       classifier = "Linear",
                       learning_approach = "HR") # Could be ERM either


# In[508]:


features = features.iloc[:, 8:]


# In[509]:


features


# In[510]:


θ, obj_value = model.fit(ξ = (np.matrix(features), np.array(diagnosis)))


# In[511]:


θ

