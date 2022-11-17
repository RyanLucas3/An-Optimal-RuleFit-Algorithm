
import pandas as pd
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
    cat_col = col[col.index == 'categorical']
    cat_col = cat_col[0].tolist()
    dummy_col = pd.get_dummies(data=df, columns=cat_col)
    add_col = dummy_col.shape[1] - df.shape[1]
    if (add_col < df.shape[0] *0.3) & (dummy_col.shape[1] <  df.shape[0]):
        df = dummy_col
    else:
        df.drop(cat_col, axis=1, inplace=True)
    return df