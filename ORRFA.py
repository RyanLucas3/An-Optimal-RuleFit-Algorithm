from sklearn.tree import _tree
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
def gen_statement(path, data, names):

    nodes_on_path = path.split("+")
    statement = ""

    for node_num, node in enumerate(nodes_on_path):

        if node_num > 0:
            prev_node = nodes_on_path[node_num - 1]
        else:
            prev_node = -10

        feature = data['lnr']['tree_']['nodes'][int(
                node)-1]['split_mixed']['parallel_split']['feature']
        threshold = data['lnr']['tree_']['nodes'][int(
                node)-1]['split_mixed']['parallel_split']['threshold']

        if int(node) == 1 and int(nodes_on_path[1]) == 2:
            equality = "<"

        elif int(node) == int(prev_node) + 1:
                equality = "<"

        else:
                equality = ">="

        if feature != 0:
                statement += f"(features['{names[feature-1]}'] {equality} {threshold})"

        # else:
        #         statement += f"({0} {equality} {threshold})"

        if int(node) != max([int(i) for i in nodes_on_path]):
                statement += " & "

    if statement[-2] == "&":
        statement = statement[:-3]

    return statement

def gen_paths(data):

    paths = []

    for child in [1]:

        node_1 = data['lnr']['tree_']['nodes'][child-1]
        node_id1 = node_1['id']

        children = sorted([node_1['upper_child'], node_1['lower_child']])

        if children[0] == -2:
            id = f"{node_id1}"
            paths.append(id)
            break

        else:


            for child in children:
                node_2 = data['lnr']['tree_']['nodes'][child-1]
                node_id2 = node_2['id']
                children = sorted([node_2['upper_child'], node_2['lower_child']])

                if children[0] == -2:
                    id = f"{node_id1}+" + f"{node_id2}"
                    paths.append(id)

                else:

                    for child in children:

                        node_3 = data['lnr']['tree_']['nodes'][child-1]
                        node_id3 = node_3['id']
                        children = sorted(
                            [node_3['upper_child'], node_3['lower_child']])

                        if children[0] == -2:
                            id = f"{node_id1}+" + f"{node_id2}+" + f"{node_id3}"
                            paths.append(id)

                        else:

                            for child in children:

                                node_4 = data['lnr']['tree_']['nodes'][child-1]
                                node_id4 = node_4['id']
                                children = sorted(
                                    [node_4['upper_child'], node_4['lower_child']])

                                if children[0] == -2:
                                    id = f"{node_id1}+" + f"{node_id2}+" + \
                                        f"{node_id3}+" + f"{node_id4}"
                                    paths.append(id)

                                else:

                                    for child in children:

                                        node_5 = data['lnr']['tree_']['nodes'][child - 1]
                                        node_id5 = node_5['id']
                                        children = sorted(
                                            [node_5['upper_child'], node_5['lower_child']])

                                        if children[0] == -2:
                                            id = f"{node_id1}+" + f"{node_id2}+" + \
                                                 f"{node_id3}+" + f"{node_id4}+" + f"{node_id5}"

                                            paths.append(id)

                                        else:

                                            for child in children:

                                                node_6 = data['lnr']['tree_']['nodes'][child - 1]
                                                node_id6 = node_6['id']
                                                children = sorted(
                                                    [node_6['upper_child'], node_6['lower_child']])

                                                if children[0] == -2:
                                                    id = f"{node_id1}+" + f"{node_id2}+" + \
                                                         f"{node_id3}+" + f"{node_id4}+" + f"{node_id5}+" + f"{node_id6}"
                                                    paths.append(id)


    if '1' in paths:
        paths.remove('1')

    return paths

def reg_results(generated_features, features, diagnosis, features_test, diagnosis_test, gen = True, use_features = True):

    if not use_features:
        features_to_use = generated_features

    elif gen and use_features:
        features_to_use = pd.concat([features, generated_features], axis = 1)

    if "Intercept" not in features_to_use.columns:
        features_to_use.insert(loc = 0, column = 'Intercept', value = [1 for i in range(len(features))])

    reg = linear_model.Ridge(alpha = 1).fit(features_to_use.copy()
    , diagnosis)
    predictions_baseline = reg.predict(features_to_use.copy())

    SSE_model = sum((diagnosis - predictions_baseline)**2)
    SSE_mean = sum((diagnosis - diagnosis.mean())**2)

    IS_R2 = 1 - SSE_model/SSE_mean

    if not use_features:
        features_test_for_use = gen_test_features(features_test, generated_features, keep_features = False)

    elif gen:
        features_test_for_use = gen_test_features(features_test, generated_features)

    if "Intercept" not in features_test_for_use.columns:
        features_test_for_use.insert(loc = 0, column = 'Intercept', value = [1 for i in range(len(features_test))])

    predictions_test = reg.predict(features_test_for_use)

    SSE_model_test = sum((diagnosis_test - predictions_test)**2)
    SSE_mean_test = sum((diagnosis_test - diagnosis.mean())**2)

    OOS_R2 = 1 - SSE_model_test/SSE_mean_test

    return IS_R2, OOS_R2, predictions_test


def gen_subpaths(paths):

    sub_paths = []

    for path in paths:

        blank_path = ""
        path_list = path.split("+")

        for i in path_list:

            path_to_append = blank_path + f"{i}"
            blank_path += f"{i} + "

            sub_paths.append(path_to_append)

    return [i for i in sub_paths if i != '1']


def gen_features(sub_paths, data, features, names):
    new_features = features.copy()
    for path in sub_paths:
        statement = gen_statement(path, data, names)
        new_features = gen_feature(statement, new_features)
    return new_features.iloc[:, len(names):]

def gen_feature(statement, new_features):
    if statement != '(0 < 0.0)' and statement != '(0 >= 0.0)':

        index_feature = new_features.loc[eval(statement)].index

        new_feature = [1 if i in index_feature else 0 for i in range(len(features))]

        if sum(new_feature) >= 1:
            new_features[statement] = new_feature

    return new_features

def do_everything(jason, features, orig_features_size, do_sub_paths = False):
    paths = gen_paths(jason)
    sub_paths = gen_subpaths(paths)
    if do_sub_paths:

        rule_features = gen_features(
            sub_paths, jason, features, orig_features_size)

    else:
        rule_features = gen_features(
            paths, jason, features, orig_features_size)

    return rule_features

def data_processing(dataset):
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month
    dataset['day'] = dataset['date'].dt.day
    dataset = dataset.drop(["date","day",'id', 'zipcode'],axis=1)
    pd.get_dummies(dataset, columns=['month'])

    features = dataset.copy()
    features = features.drop('price', axis = 1)
    diagnosis = dataset.loc[:, "price"]

    features_train, features_test, diagnosis_train, diagnosis_test = train_test_split(features, diagnosis, test_size = 0.3, shuffle=True)

    orig_columns = features.copy().columns

    sc = StandardScaler()
    features_train = pd.DataFrame(sc.fit_transform(features_train))
    features_train.columns = orig_columns

    features_test = pd.DataFrame(sc.fit_transform(features_test))
    features_test.columns = orig_columns

    return features_train, diagnosis_train, features_test, diagnosis_test


def get_rules(tree, feature_names, class_names,features):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    name_data = "features"
    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name_data}['{name}'] < {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name_data}['{name}'] >= {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = ""
        for p in path[:-1]:
            if rule != "":
                rule += " & "
            rule += str(p)
        rules += [rule]

    return rules
