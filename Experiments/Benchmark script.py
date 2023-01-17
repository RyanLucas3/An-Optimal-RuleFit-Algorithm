from pauls_functions_advanced_v3 import *
from experiment_functions import *
import pandas as pd
from pmlb import fetch_data, classification_dataset_names
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
#%%
from tqdm.auto import tqdm
from joblib import Parallel

class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
#%%
def get_feature_type(x, include_binary=False):
    x.dropna(inplace=True)
    if not check_if_all_integers(x):
        return 'continuous'
    else:
        if x.nunique() > 10:
            return 'continuous'
        if include_binary:
            if x.nunique() == 2:
                return 'binary'
        return 'categorical'

def get_target_type(x, include_binary=False):
    x.dropna(inplace=True)
    if x.dtype=='float64':
        return 'continuous'
    elif x.dtype=='int64':
        if include_binary:
            if x.nunique() == 2:
                return 'binary'
        return 'categorical'
    else:
        raise ValueError("Error getting type")

def check_if_all_integers(x):
    "check a pandas.Series is made of all integers."
    return all(float(i).is_integer() for i in x.unique())
def corr_data_for(df):
    TARGET_NAME = 'target'
    feat_names = [col for col in df.columns if col!=TARGET_NAME]
    types = [get_feature_type(df[col], include_binary=True) for col in feat_names]
    col = pd.DataFrame(feat_names,types)
    num_col = col[col.index == 'continuous']
    bin_col = col[col.index == 'binary']
    cat_col = col[col.index == 'categorical']
    cat_col = cat_col[0].tolist()
    dummy_col = pd.get_dummies(data=df, columns=cat_col)
    add_col = dummy_col.shape[1] - df.shape[1]
    if (add_col < df.shape[0] *0.3) & (dummy_col.shape[1] <  df.shape[0]):
        df = dummy_col
        df.columns = df.columns.str.replace('.','_',regex=True)
    else:
        del df
        df = pd.DataFrame()
    return df, num_col, bin_col, cat_col
#%%
# for data in classification_dataset_names:
#     data = fetch_data(data)
#     print(data.shape)
#%%
classification_dataset_names = classification_dataset_names[1:10]
#%%
def experimentation(classification_dataset):
    iters=5
    res_rul = {}
    sc = StandardScaler()
    names = ['Reg-CART','CART','ORT','OCT','ORT-H','OCT-H','ORT+ORT-H','OCT+OCT-H']
    df = fetch_data(classification_dataset)
    if df.shape[0] > 50000:
        return
    if df.shape[1] > 100:
        return
    df, num_col, bin_col, cat_col = corr_data_for(df)
    if df.empty:
        return
    y = df['target']
    X = df.loc[:, df.columns != 'target']
    #performance_by_iter = pd.DataFrame(columns = ["Logistic Regression", "CART_rules", "OCT_rules", "OCTH_rules", "CART_rules_and_features", "OCT_rules_and_features", "OCTH_rules_and_features"], index = np.arange(0, iters))
    print(color.BOLD + '\n\n    ----------------------------------------- {} -----------------------------------------'.format(classification_dataset) + color.END)
    rows_data, columns_data = X.shape
    print('Dataset Information')
    print('Rows:',rows_data,)
    print('Columns:',columns_data)
    print('Number of classes:',y.nunique())
    print('Continous columns:', len(num_col))
    print('Binary columns:', len(bin_col))
    print('Categorical columns:',len(cat_col))
    print('-------------------------------------------------')
    for it in range(iters):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = it, stratify=y)
        X_col = X_train.columns
        col_len = len(X_col)
        X_test.name = "X_test"
        X_train.name = "X_train"
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        X_train = pd.DataFrame(X_train,columns=X_col)
        X_test = pd.DataFrame(X_test,columns=X_col)
        factors = [round(col_len*0.5),col_len,round(col_len*1.2),round(col_len*1.4),round(col_len*1.6),round(col_len*1.8),round(col_len*2),round(col_len*2.5),round(col_len*3)]
        factors_name = [0.5,1,1.2,1.4,1.6,1.8,2,2.5,3]



        models, performance = generate_tree(X_train, y_train, X_test, y_test, n_num=1, feat_size=len(X.columns),  max_iter_hy=2,sub_paths=False,depth_grid=range(1,7), depth_grid_hy=range(1,3), complexity_bi = 0.001, complexity_hy=0.001,  Reg_CART=False, ORT=False, ORT_H=False, Clas_CART=True, OCT=True, OCT_H=False)
        for perf,name in zip(performance,names):
            if not not perf:
                res_rul[(classification_dataset,name,it,1)] = sum(perf) / len(perf)

        act_name = []
        act_rules = []
        for model,name in zip(models,names):
            if not not model:
                act_name += [name]
                act_rules += [model]

        datasets = gen_train_and_test_features(act_rules ,act_name , X_train, X_test)
        for model in datasets.keys():
            print(model)
            X_train_rules_and_features, X_test_rules_and_features = datasets[model][0]
            X_train_only_rules, X_test_only_rules = datasets[model][1]
            print(len(X_train_rules_and_features.columns))
            for len_c,fac_name in zip(factors,factors_name):
                if len_c > len(X_train_only_rules.columns):
                    min_len = len(X_train_only_rules.columns)
                    min_name = 1
                else:
                    min_len = len_c
                    min_name = fac_name
                if (len_c <= X_train.shape[0]) & (len_c <= len(X_train_rules_and_features.columns)):
                    cols = SelectKBest(k=len_c).fit(X_train_rules_and_features,y_train).get_feature_names_out()
                    X_train_rules_features = X_train_rules_and_features[cols]
                    X_test_rules_features = X_test_rules_and_features[cols]

                    cols_1 = SelectKBest(k=min_len).fit(X_train_only_rules,y_train).get_feature_names_out()
                    X_train_rules = X_train_only_rules[cols_1]
                    X_test_rules = X_test_only_rules[cols_1]

                    only_rules_acc = log_regression_pipeline(X_train_rules, X_test_rules, y_train, y_test)
                    rules_and_features_acc = log_regression_pipeline(X_train_rules_features, X_test_rules_features, y_train, y_test)
                    res_rul[(classification_dataset,model + "_LG_rules",it,min_name)] = only_rules_acc
                    res_rul[(classification_dataset,model + "_LG_rules_and_features",it,fac_name)] = rules_and_features_acc

                    only_rules_acc_SVM = SVM_pipeline(X_train_rules, X_test_rules, y_train, y_test)
                    rules_and_features_acc_SVM = SVM_pipeline(X_train_rules_features, X_test_rules_features, y_train, y_test)
                    res_rul[(classification_dataset,model + "_SVM_rules",it,min_name)] = only_rules_acc_SVM
                    res_rul[(classification_dataset,model + "_SVM_rules_and_features",it,fac_name)] = rules_and_features_acc_SVM

                    only_rules_acc_NB = NB_pipeline(X_train_rules, X_test_rules, y_train, y_test)
                    rules_and_features_acc_NB = NB_pipeline(X_train_rules_features, X_test_rules_features, y_train, y_test)
                    res_rul[(classification_dataset,model + "_NB_rules",it,min_name)] = only_rules_acc_NB
                    res_rul[(classification_dataset,model + "_NB_rules_and_features",it,fac_name)] = rules_and_features_acc_NB

                    only_rules_acc_KNN = KNN_pipeline(X_train_rules, X_test_rules, y_train, y_test)
                    rules_and_features_acc_KNN = KNN_pipeline(X_train_rules_features, X_test_rules_features, y_train, y_test)
                    res_rul[(classification_dataset,model + "_KNN_rules",it,min_name)] = only_rules_acc_KNN
                    res_rul[(classification_dataset,model + "_KNN_rules_and_features",it,fac_name)] = rules_and_features_acc_KNN
                else:
                     continue

        res_rul[(classification_dataset,'Logistic_Regression',it,1)] = log_regression_pipeline(X_train, X_test, y_train, y_test)

        res_rul[(classification_dataset,"Support Vector Machine",it,1)] = SVM_pipeline(X_train, X_test, y_train, y_test)

        res_rul[(classification_dataset,"Naive Bayes",it,1)] = NB_pipeline(X_train, X_test, y_train, y_test)

        res_rul[(classification_dataset,"K-Nearest-Neighbor",it,1)] = KNN_pipeline(X_train, X_test, y_train, y_test)
        with open(classification_dataset + '.pickle', 'wb') as handle:
            pickle.dump(res_rul, handle)
    return res_rul

from joblib import delayed
from tqdm import tqdm
res_rul = ProgressParallel(n_jobs=6)(delayed(experimentation)(data) for data in classification_dataset_names)