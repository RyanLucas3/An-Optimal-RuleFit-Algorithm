import statistics
from sklearn import tree
from sklearn.tree import _tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
import json
from julia.api import Julia
jl = Julia(compiled_modules=False)
from interpretableai import iai
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_rules(tree, feature_names):
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


def generate_tree(X_train, y_train, X_test, y_test, depth, n_num, Reg_CART="True", ORT="True", Clas_CART="True", OCT="True"):
    perf_reg_cart = []
    dt_rules_reg_cart = []
    if Reg_CART == "True":
        for i in range(1, n_num+1):
            model = DecisionTreeRegressor(max_depth=depth)
            model.fit(X_train, y_train)
            perf_reg_cart += [model.score(X_test, y_test)]
            dt_rules_reg_cart += [get_rules(model, X_train.columns)]
        print("-----------------------------------------------------------------")
        print("Regression CART mean performance: ",
              statistics.mean(perf_reg_cart))
        if n_num > 1:
            print("Standard Deviation: ", statistics.stdev(perf_reg_cart))
        print("\n")

    perf_cla_cart = []
    dt_rules_cla_cart = []
    if Clas_CART == "True":
        for i in range(1, n_num+1):
            model = tree.DecisionTreeClassifier(max_depth=depth)
            model.fit(X_train, y_train)
            perf_cla_cart += [model.score(X_test, y_test)]
            dt_rules_cla_cart += [get_rules(model, X_train.columns)]
        print("-----------------------------------------------------------------")
        print("Classification CART mean performance: ",
              statistics.mean(perf_cla_cart))
        if n_num > 1:
            print("Standard Deviation: ", statistics.stdev(perf_cla_cart))
        print("\n")

    perf_reg_ort = []
    dt_rules_reg_ort = []
    if ORT == "True":
        for i in range(1, n_num+1):
            ORT = iai.GridSearch(iai.OptimalTreeClassifier(max_depth=depth))
            ORT.fit(X_train, y_train)
            model = iai.OptimalTreeClassifier(
                **ORT.get_best_params(), max_depth=depth)
            model.fit(X_train, y_train)
            perf_reg_ort += [model.score(X_test, y_test)]
            dt_rules_reg_ort = get_optimal_rules(model)
        print("-----------------------------------------------------------------")
        print("Regression ORT mean performance: ",
              statistics.mean(perf_reg_ort))
        if n_num > 1:
            print("Standard Deviation: ", statistics.stdev(perf_reg_ort))
        print("\n")

    perf_cla_oct = []
    dt_rules_cla_oct = []
    if OCT == "True":
        for i in range(1, n_num+1):
            OCT = iai.GridSearch(iai.OptimalTreeClassifier(max_depth=depth))
            OCT.fit(X_train, y_train)
            model = iai.OptimalTreeClassifier(
                **OCT.get_best_params(), max_depth=depth)
            model.fit(X_train, y_train)
            perf_cla_oct += [model.score(X_test, y_test)]
            dt_rules_cla_oct = get_optimal_rules(model)
        print("-----------------------------------------------------------------")
        print("Regression ORT mean performance: ",
              statistics.mean(perf_cla_oct))
        if n_num > 1:
            print("Standard Deviation: ", statistics.stdev(perf_cla_oct))
        print("\n")

    return dt_rules_cla_cart, dt_rules_reg_cart, dt_rules_reg_ort, dt_rules_cla_oct


def get_optimal_rules(model):
    rules = []
    for i in range(1, model.get_num_nodes()+1):
        if model.is_leaf(i):
            t = i
            name_data = "feature"
            rule_part = []
            while True:
                if t == 1:
                    condition = "st_inequality" if k == 2 else "inequality"
                elif t != i:
                    model.get_parent(t)
                    condition = "st_inequality" if k == model.get_lower_child(
                        t) else "inequality"
                if model.is_parallel_split(t):
                    name = model.get_split_feature(t)
                    threshold = model.get_split_threshold(t)
                    if condition == "st_inequality":
                        rule_part += [f"({name_data}['{name}'] < {np.round(threshold, 3)})"]
                    else:
                        rule_part += [f"({name_data}['{name}'] >= {np.round(threshold, 3)})"]

                elif model.is_mixed_parallel_split(t):
                    threshold = model.get_split_threshold(model, t)
                    categories = model.get_split_categories(model, t)
                    # goes_lower = isa(x, Real) ? x < threshold : categories[x]
                if t == 1:
                    rule = ""
                    rule_part.reverse()
                    for p in rule_part:
                        if rule != "":
                            rule += " & "
                        rule += str(p)
                    rules += [rule]
                    break
                k = t

                t = model.get_parent(t)
    return rules
