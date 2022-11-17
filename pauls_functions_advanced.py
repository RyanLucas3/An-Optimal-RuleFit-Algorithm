import statistics
from sklearn import tree
from sklearn.tree import _tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
#import json
from interpretableai import iai
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def get_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    name_data = "feature"

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


def get_optimal_rules(model):
    rules = []
    for i in range(1,model.get_num_nodes()+1):
        if model.is_leaf(i):
            if i == 1:
                print(model, "is of depth 1 - cannot give out rules")
            else:
                t = i
                name_data = "feature"
                rule_part = []
                while True:
                    if t == 1:
                        condition = "st_inequality" if k == 2 else "inequality"
                    elif t != i:
                        model.get_parent(t)
                        condition = "st_inequality" if k == model.get_lower_child(t) else "inequality"

                    if model.is_hyperplane_split(t):
                        numeric_weights, categoric_weights = model.get_split_weights(t)
                        split_value = []
                        threshold = model.get_split_threshold(t)
                        for (feature, weight) in numeric_weights.items():
                            split_value += [f"{weight} * {name_data}['{feature}']"]
                            "+".join(split_value)
                            split = "+".join(split_value)
                        if condition == "st_inequality":
                            rule_part += [f"({split} < {np.round(threshold, 3)})"]
                        else:
                            rule_part += [f"({split} >= {np.round(threshold, 3)})"]

                    if model.is_parallel_split(t):

                        name = model.get_split_feature(t)
                        threshold = model.get_split_threshold(t)
                        if condition == "st_inequality":
                            rule_part += [f"({name_data}['{name}'] < {np.round(threshold, 3)})"]
                        else:
                            rule_part += [f"({name_data}['{name}'] >= {np.round(threshold, 3)})"]

                    elif model.is_mixed_parallel_split(t):
                        threshold = model.get_split_threshold(model,t)
                        categories = model.get_split_categories(model, t)
                        #goes_lower = isa(x, Real) ? x < threshold : categories[x]
                    if t == 1:
                        rule = ""
                        rule_part.reverse()
                        for p in rule_part:
                            if rule != "":
                                rule += " & "
                            rule += str(p)
                        rules += [rule]
                        break
                    k=t

                    t = model.get_parent(t)
    return rules


def gen_rules(df, rules):

    df = df.copy()

    for rule in rules:
        rule_copy = rule[:-1] + ")"
        rule = rule.replace("feature", "df")

        filter_idx = df.loc[eval(rule)].index.values

        new_feature = []

        for i in df.index.values:
            if i in filter_idx:
                new_feature.append(1)
            else:
                new_feature.append(0)

        df[rule_copy] = new_feature

    return df

def generate_tree(X_train, y_train, X_test , y_test , n_num, feat_size,
                  max_iter_hy, depth_bi=False, depth_hy=False, complexity_bi=False,
                  complexity_hy=False, depth_grid_bi=False, depth_grid_hy=False,
                  Reg_CART=False, ORT=False, ORT_H=False, Clas_CART=False, OCT=False, OCT_H=False):

    if depth_grid_bi != False and depth_bi > 0:
        raise Exception("Both grid search and normal depth are specified - eat less eggs")

    if depth_grid_hy != False and depth_hy > 0:
        raise Exception("Both grid search and normal depth are specified - the EV is always zero")

    perf_reg_cart = []
    rules_reg_cart = []
    perf_cla_cart = []
    rules_cla_cart = []
    perf_reg_ort = []
    rules_reg_ort = []
    perf_cla_oct = []
    rules_cla_oct = []
    perf_reg_ort_h = []
    rules_reg_ort_h = []
    perf_cla_oct_h = []
    rules_cla_oct_h = []

    for i in range(1, n_num + 1):
        if feat_size > 0:
            index = X_train.columns.to_series().sample(feat_size)
            X_train_sample = X_train[index]
            X_test_sample = X_test[index]
        else:
            X_train_sample = X_train
            X_test_sample = X_test

        if Reg_CART == True:
            if complexity_bi == False and depth_bi == False:
                model = iai.GridSearch(iai.OptimalTreeRegressor(localsearch=False,criterion='mse'),max_depth=depth_grid_bi)
                model.fit(X_train_sample, y_train)
            if complexity_bi == False and depth_bi > 0:
                model = iai.GridSearch(iai.OptimalTreeRegressor(max_depth = depth_bi, localsearch=False,criterion='mse'))
                model.fit(X_train_sample, y_train)
            if complexity_bi > 0  and depth_bi == False:
                model = iai.GridSearch(iai.OptimalTreeRegressor(localsearch=False,criterion='mse'),cp=complexity_bi,max_depth=depth_grid_bi)
                model.fit(X_train_sample, y_train)
            if complexity_bi > 0 and depth_bi > 0:
                model = iai.OptimalTreeRegressor(max_depth=depth_bi,cp=complexity_bi, localsearch=False,criterion='mse')
                model.fit(X_train_sample, y_train)
            perf_reg_cart += [model.score(X_test_sample , y_test)]
            rules_reg_cart +=  get_optimal_rules(model)
            print("Regression CART mean performance: ",model.score(X_test_sample , y_test))
            print("\n")
        else:
            perf_reg_cart.append(np.nan)


        if Clas_CART == True:
            if complexity_bi == False and depth_bi == False:
                model = iai.GridSearch(iai.OptimalTreeClassifier(localsearch=False,criterion='gini'),max_depth=depth_grid_bi)
                model.fit(X_train_sample, y_train)
            if complexity_bi == False and depth_bi > 0 :
                model = iai.GridSearch(iai.OptimalTreeClassifier(max_depth = depth_bi,localsearch=False,criterion='gini'))
                model.fit(X_train_sample, y_train)
            if complexity_bi > 0  and depth_bi == False:
                model = iai.GridSearch(iai.OptimalTreeClassifier( localsearch=False,criterion='gini'),cp=complexity_bi,max_depth=depth_grid_bi)
                model.fit(X_train_sample, y_train)
            if complexity_bi > 0 and depth_bi > 0:
                model = iai.OptimalTreeClassifier(max_depth=depth_bi,cp=complexity_bi,localsearch=False,criterion='gini')
                model.fit(X_train_sample, y_train)
            perf_cla_cart += [model.score(X_test_sample , y_test,criterion='misclassification')]
            rules_cla_cart += get_optimal_rules(model)
            print("Classification CART mean performance: ",model.score(X_test_sample , y_test,criterion='misclassification'))
            print("\n")
        else:
            perf_cla_cart.append(np.nan)


        if ORT == True:
            if complexity_bi == False and depth_bi == False:
                model = iai.GridSearch(iai.OptimalTreeRegressor(),max_depth=depth_grid_bi)
                model.fit(X_train_sample, y_train)
            if complexity_bi == False and depth_bi > 0:
                model = iai.GridSearch(iai.OptimalTreeRegressor(max_depth = depth_bi))
                model.fit(X_train_sample, y_train)
            if complexity_bi > 0  and depth_bi == False:
                model = iai.GridSearch(iai.OptimalTreeRegressor(),cp=complexity_bi,max_depth=depth_grid_bi)
                model.fit(X_train_sample, y_train)
            if complexity_bi > 0 and depth_bi > 0:
                model = iai.OptimalTreeRegressor(max_depth=depth_bi,cp=complexity_bi)
                model.fit(X_train_sample, y_train)
            perf_reg_ort += [model.score(X_test_sample , y_test)]
            rules_reg_ort += get_optimal_rules(model)
            print("Regression ORT performance: ",model.score(X_test_sample , y_test))
            print("\n")
        else:
            perf_reg_ort.append(np.nan)


        if OCT == True:
            if complexity_bi == False and depth_bi == False:
                model = iai.GridSearch(iai.OptimalTreeClassifier(),max_depth=depth_grid_bi)
                model.fit(X_train_sample, y_train)
            if complexity_bi == False and depth_bi > 0 :
                model = iai.GridSearch(iai.OptimalTreeClassifier(max_depth = depth_bi))
                model.fit(X_train_sample, y_train)
            if complexity_bi > 0  and depth_bi == False:
                model = iai.GridSearch(iai.OptimalTreeClassifier(),cp=complexity_bi,max_depth=depth_grid_bi)
                model.fit(X_train_sample, y_train)
            if complexity_bi > 0 and depth_bi > 0:
                model = iai.OptimalTreeClassifier(max_depth=depth_bi,cp=complexity_bi)
                model.fit(X_train_sample, y_train)
            perf_cla_oct += [model.score(X_test_sample , y_test,criterion='misclassification')]
            rules_cla_oct += get_optimal_rules(model)
            print("Classification OCT performance: ",model.score(X_test_sample , y_test,criterion='misclassification'))
            print("\n")
        else:
            perf_cla_oct.append(np.nan)



        if ORT_H == True:
            if i <= max_iter_hy:
                if complexity_bi == False and depth_bi == False:
                    model = iai.GridSearch(iai.OptimalTreeRegressor(hyperplane_config={'sparsity': 'all'}), max_depth=depth_grid_hy)
                    model.fit(X_train_sample, y_train)
                if complexity_hy == False and depth_hy > 0:
                    model = iai.GridSearch(iai.OptimalTreeRegressor(max_depth=depth_hy,hyperplane_config={'sparsity': 'all'}))
                    model.fit(X_train_sample, y_train)
                if complexity_hy > 0 and depth_hy == False:
                    model = iai.GridSearch(iai.OptimalTreeRegressor( hyperplane_config={'sparsity': 'all'}),cp=complexity_hy, max_depth=depth_grid_hy)
                    model.fit(X_train_sample, y_train)
                if complexity_hy > 0 and depth_hy > 0:
                    model = iai.OptimalTreeRegressor(max_depth=depth_hy, cp=complexity_hy, hyperplane_config={'sparsity': 'all'})
                    model.fit(X_train_sample, y_train)
                perf_reg_ort_h += [model.score(X_test_sample , y_test)]
                rules_reg_ort_h += get_optimal_rules(model)
                print("Regression ORT_H performance: ",model.score(X_test_sample , y_test))
                print("\n")
            else:
                perf_reg_ort_h.append(np.nan)
        else:
            perf_reg_ort_h.append(np.nan)

        if OCT_H == True:
            if i <= max_iter_hy:
                if complexity_bi == False and depth_bi == False:
                    model = iai.GridSearch(iai.OptimalTreeClassifier(hyperplane_config={'sparsity': 'all'}), max_depth=depth_grid_hy)
                    model.fit(X_train_sample, y_train)
                if complexity_hy == False and depth_hy > 0:
                    model = iai.GridSearch(iai.OptimalTreeClassifier(max_depth=depth_hy, hyperplane_config={'sparsity': 'all'}))
                    model.fit(X_train_sample, y_train)
                if complexity_hy > 0 and depth_hy == False:
                    model = iai.GridSearch(iai.OptimalTreeClassifier( hyperplane_config={'sparsity': 'all'}),cp=complexity_hy, max_depth=depth_grid_hy)
                    model.fit(X_train_sample, y_train)
                if complexity_hy > 0 and depth_hy > 0:
                    model = iai.OptimalTreeClassifier(max_depth=depth_hy, cp=complexity_hy, hyperplane_config={'sparsity': 'all'})
                    model.fit(X_train_sample, y_train)
                perf_cla_oct_h += [model.score(X_test_sample , y_test,criterion='misclassification')]
                rules_cla_oct_h += get_optimal_rules(model)
                print("Classification OCT_H performance: ",model.score(X_test_sample , y_test,criterion='misclassification'))
                print("\n")
            else:
                perf_cla_oct_h.append(np.nan)
        else:
            perf_cla_oct_h.append(np.nan)

    performance = pd.DataFrame(zip(perf_reg_cart, perf_cla_cart, perf_reg_ort, perf_cla_oct, perf_reg_ort_h, perf_cla_oct_h),columns = ['Reg CART', 'Cla CART','ORT','OCT','ORT-H','OCT-H'])
    models = [rules_reg_cart, rules_cla_cart, rules_reg_ort, rules_cla_oct, rules_reg_ort_h, rules_cla_oct_h]

    return  models, performance


def linear_regression_pipeline(X_train, X_test, y_train, y_test):

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model.score(X_test, y_test)


def log_regression_pipeline(X_train, X_test, y_train, y_test):

    model = LogisticRegression(random_state=0,solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)

    return accuracy_score(y_test, model.predict(X_test))


def gen_train_and_test_features(rules, names, X_train, X_test):

    train_and_test_sets = {}

    for i, rules in enumerate(rules):

        X_train_rules = gen_rules(X_train, rules)
        X_test_rules = gen_rules(X_test, rules)
        X_train_only_rules = X_train_rules.loc[:, X_train_rules.columns.str.contains("feature")]
        X_test_only_rules = X_test_rules.loc[:, X_test_rules.columns.str.contains("feature")]

        train_and_test_sets[names[i]] = [[X_train_rules, X_test_rules], [X_train_only_rules, X_test_only_rules]]

    return train_and_test_sets


