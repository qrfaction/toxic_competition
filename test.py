import pandas as pd
import numpy as np
PATH='data/'

def post_deal():
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    a=pd.read_csv('baseline.csv.gz')
    # a[list_classes]=np.exp(np.log(a[list_classes]) -0.5)
    a[list_classes]=a[list_classes]**1.3

    a.to_csv("test.csv.gz", index=False, compression='gzip')

def cal_mean():
    from sklearn.cross_validation import KFold

    labels= pd.read_csv(PATH+'labels.csv')

    print(labels.describe())

    kf = KFold(len(labels), n_folds=6, shuffle=False)
    print(list(kf)[3])
    for i, (train_index, valid_index) in enumerate(kf):
        print(i,'----------------')
        print(labels.iloc[valid_index].describe())
        print(labels.iloc[train_index].describe())

# cal_mean()
post_deal()