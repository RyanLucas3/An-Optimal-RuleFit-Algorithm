#!/usr/bin/env python
# coding: utf-8
# author: Bo Tang

from collections import namedtuple
import numpy as np
from scipy import stats
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from sklearn import tree
from sklearn.linear_model import LinearRegression

def convert_dataframe_to_matrix(data):
    if isinstance(data, pd.DataFrame):
        matrix = data.to_numpy()
        return matrix

    elif isinstance(data, pd.Series):
        array = data.to_numpy()
        return array
    
    else:
        return data
    
class optimalDecisionTreeClassifier:
    """
    optimal classification tree
    """
    def __init__(self, max_depth=3, min_samples_split=2, alpha=0, warmstart=True, timelimit=600, output=True):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.alpha = alpha
        self.warmstart = warmstart
        self.timelimit = timelimit
        self.output = output
        self.trained = False
        self.optgap = None

        # node index
        self.n_index = [i+1 for i in range(2 ** (self.max_depth + 1) - 1)]
        self.b_index = self.n_index[:-2**self.max_depth] # branch nodes
        self.l_index = self.n_index[-2**self.max_depth:] # leaf nodes

    def fit(self, x, y, scaled = False):
        """
        fit training data
        """
        # data size
        self.n, self.p = x.shape
        if self.output:
            print('Training data include {} instances, {} features.'.format(self.n,self.p))

        # labels
        self.labels = np.unique(y)

        # scale data

        x, y = convert_dataframe_to_matrix(x), convert_dataframe_to_matrix(y)

        if not scaled:
            self.scales = np.max(x, axis=0)
            self.scales[self.scales == 0] = 1
            x = x/self.scales

        # solve MIP
        m, a, b, c, d, l, beta, gamma = self._buildMIP(x, y)
        if self.warmstart:
            self._setStart(x, y, a, d, beta)

        m.optimize()
        self.optgap = m.MIPGap

        # get parameters
        self._a = {ind:a[ind].x for ind in a}
        self._b = {ind:b[ind].x for ind in b}
        self._c = {ind:c[ind].x for ind in c}
        self._d = {ind:d[ind].x for ind in d}
        self.beta = {ind:beta[ind].x for ind in beta}
        self.gamma = {ind:gamma[ind].x for ind in gamma}

        self.trained = True

    def predict(self, x):
        """
        model prediction
        """
        if not self.trained:
            raise AssertionError('This optimalDecisionTreeClassifier instance is not fitted yet.')

        # leaf label
        labelmap = {}
        for t in self.l_index:
            for k in self.labels:
                if self._c[k,t] >= 1e-2:
                    labelmap[t] = k

        y_pred = []
        for xi in x/self.scales:
            t = 1
            while t not in self.l_index:
                right = (sum([self._a[j,t] * xi[j] for j in range(self.p)]) + 1e-9 >= self._b[t])
                if right:
                    t = 2 * t + 1
                else:
                    t = 2 * t
            # label
            y_pred.append(labelmap[t])

        return np.array(y_pred)

    def _buildMIP(self, x, y):
        """
        build MIP formulation for Optimal Decision Tree
        """
        # create a model
        m = gp.Model('m')

        # output
        m.Params.outputFlag = self.output
        m.Params.LogToConsole = self.output
        # time limit
        m.Params.timelimit = self.timelimit
        # parallel
        m.params.threads = 0

        # model sense
        m.modelSense = GRB.MINIMIZE

        # variables
        a = m.addVars(self.p, self.b_index, vtype=GRB.BINARY, name='a') # splitting feature
        b = m.addVars(self.b_index, vtype=GRB.CONTINUOUS, name='b') # splitting threshold
        c = m.addVars(self.labels, self.l_index, vtype=GRB.BINARY, name='c') # node prediction
        d = m.addVars(self.b_index, vtype=GRB.BINARY, name='d') # splitting option
        z = m.addVars(self.n, self.l_index, vtype=GRB.BINARY, name='z') # leaf node assignment
        l = m.addVars(self.l_index, vtype=GRB.BINARY, name='l') # leaf node activation
        beta = m.addVars(self.p, vtype=GRB.CONTINUOUS, name='beta') # beta
        gamma = m.addVars(self.l_index, vtype=GRB.CONTINUOUS, name='gamma') # beta
        lamb = m.addVars(self.n, vtype=GRB.CONTINUOUS, name='lambda') # beta
        # L = m.addVars(self.l_index, vtype=GRB.CONTINUOUS, name='L') # leaf node misclassified
        M = m.addVars(self.labels, self.l_index, vtype=GRB.CONTINUOUS, name='M') # leaf node samples with label
        N = m.addVars(self.l_index, vtype=GRB.CONTINUOUS, name='N') # leaf node samples
        aux = m.addVars(self.n, vtype = GRB.CONTINUOUS, name = 'aux')

        # calculate baseline accuracy
        baseline = self._calBaseline(y)

        # calculate minimum distance
        min_dis = self._calMinDist(x)
        
        objExp = gp.QuadExpr()

        # objective function
        for i in range(self.n):
            var = y[i] - gp.quicksum(x[i, p] * beta[p] for p in range(self.p)) - lamb[i]
            objExp.add(var * var) 
            
            m.addConstr(lamb[i] == gp.quicksum(gamma[t]*z[i, t] for t in self.l_index))

        objExp.add(self.alpha * d.sum())
        m.setObjective(objExp)

        m.addConstrs(z.sum('*', t) == N[t] for t in self.l_index)

        # (13) and (14)
        for t in self.l_index:
            left = (t % 2 == 0)
            ta = t // 2
            while ta != 0:
                if left:
                    m.addConstrs(gp.quicksum(a[j,ta] * (x[i,j] + min_dis[j]) for j in range(self.p))
                                 +
                                 (1 + np.max(min_dis)) * (1 - d[ta])
                                 <=
                                 b[ta] + (1 + np.max(min_dis)) * (1 - z[i,t])
                                 for i in range(self.n))
                else:
                    m.addConstrs(gp.quicksum(a[j,ta] * x[i,j] for j in range(self.p))
                                 >=
                                 b[ta] - (1 - z[i,t])
                                 for i in range(self.n))
                left = (ta % 2 == 0)
                ta //= 2
        # (8)
        m.addConstrs(z.sum(i, '*') == 1 for i in range(self.n))
        # (6)
        m.addConstrs(z[i,t] <= l[t] for t in self.l_index for i in range(self.n))
        # (7)
        m.addConstrs(z.sum('*', t) >= self.min_samples_split * l[t] for t in self.l_index)
        # (2)
        m.addConstrs(a.sum('*', t) == d[t] for t in self.b_index)
        # (3)
        m.addConstrs(b[t] <= d[t] for t in self.b_index)
        # (5)
        m.addConstrs(d[t] <= d[t//2] for t in self.b_index if t != 1)

        return m, a, b, c, d, l, beta, gamma

    @staticmethod
    def _calBaseline(y):
        """
        obtain baseline accuracy by simply predicting the most popular class
        """
        mode = stats.mode(y)[0][0]
        return np.sum(y == mode)

    @staticmethod
    def _calMinDist(x):
        """
        get the smallest non-zero distance of features
        """
        min_dis = []
        for j in range(x.shape[1]):
            xj = x[:,j]
            # drop duplicates
            xj = np.unique(xj)
            # sort
            xj = np.sort(xj)[::-1]
            # distance
            dis = [1]
            for i in range(len(xj)-1):
                dis.append(xj[i] - xj[i+1])
            # min distance
            min_dis.append(np.min(dis) if np.min(dis) else 1)
        return min_dis

    def _setStart(self, x, y, a, d, beta):
        """
        set warm start from CART for regression tree
        """

        reg_model = LinearRegression()
        reg_model.fit(x, y)

        # Warm-start beta with regression coefficients
        beta.start = reg_model.coef_

        # Train with CART
        if self.min_samples_split > 1:
            clf = tree.DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        else:
            clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
        clf.fit(x, y)

        # Get splitting rules
        rules = self._getRules(clf)

        # Fix branch node
        for t in self.b_index:
            # Not split
            if rules[t].feat is None or rules[t].feat == tree._tree.TREE_UNDEFINED:
                d[t].start = 0
                for f in range(self.p):
                    a[f, t].start = 0
            # Split
            else:
                d[t].start = 1
                for f in range(self.p):
                    if f == int(rules[t].feat):
                        a[f, t].start = 1
                    else:
                        a[f, t].start = 0


    def _getRules(self, clf):
        """
        get splitting rules
        """
        # node index map
        node_map = {1:0}
        for t in self.b_index:
            # terminal
            node_map[2*t] = -1
            node_map[2*t+1] = -1
            # left
            l = clf.tree_.children_left[node_map[t]]
            node_map[2*t] = l
            # right
            r = clf.tree_.children_right[node_map[t]]
            node_map[2*t+1] = r

        # rules
        rule = namedtuple('Rules', ('feat', 'threshold', 'value'))
        rules = {}
        # branch nodes
        for t in self.b_index:
            i = node_map[t]
            if i == -1:
                r = rule(None, None, None)
            else:
                r = rule(clf.tree_.feature[i], clf.tree_.threshold[i], clf.tree_.value[i,0])
            rules[t] = r
        # leaf nodes
        for t in self.l_index:
            i = node_map[t]
            if i == -1:
                r = rule(None, None, None)
            else:
                r = rule(None, None, clf.tree_.value[i,0])
            rules[t] = r

        return rules
