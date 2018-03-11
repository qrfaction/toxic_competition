import pandas as pd
import numpy as np
from tqdm import tqdm
PATH='data/'

def cal_mean():
    from sklearn.cross_validation import KFold
    a=pd.read_csv('data/labels.csv')
    # a[list_classes]=np.exp(np.log(a[list_classes]) -0.5)
    kf = KFold(a.shape[0], n_folds=6, shuffle=False)
    for train_index,valid_index in kf:
        tr_set = a.iloc[train_index]
        valid_set = a.iloc[valid_index]
        print(tr_set.describe())
        print(valid_set.describe())

def get_corr():
    usecol = [
        'toxicity_score_level',
        'quoting_attack_level',
        'recipient_attack_level',
        'third_party_attack_level',
        'other_attack_level',
        'toxicity_level',
        'attack_level',
    ]
    dataset = pd.read_csv(PATH+'clean_train.csv',usecols=usecol)
    print(dataset.corr())

def test():
    a = pd.read_csv('data/labels.csv')
    a= a.loc[a['toxic']==0,"severe_toxic"]
    print(a)
    print(a.sum())

def post_deal():
    output = pd.read_csv('baseline.csv.gz')

    def deal(row):
        if row["severe_toxic"]>0.5 and row["toxic"] <0.5 :
            print(111111)
            row['severe_toxic'] = 0
        return row
    output = output.apply(deal,axis=1)
    output.to_csv('output.csv.gz', index=False, compression='gzip')

def bagging():
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    PATH = 'results/'
    file_weight = {
        # 'corr_blend.csv.gz':10,
        # 'my56.csv.gz':10,
        # "celoss_56.csv.gz":2,
        "celoss2_55.csv.gz":1,


        # "focal62.csv.gz":6,"mybest63.csv.gz":8,    9865
        # "62_63To65.csv.gz":7,"one_more_blend_65.csv.gz":5,   9874
        "62_63To65.csv.gz":5,
        "blend_it_all_68.csv.gz":5,
    }
    output = pd.read_csv(PATH+'baseline.csv.gz')
    output[list_classes] = 0
    norm = 0
    for file,weight in file_weight.items():
        result = pd.read_csv(PATH+file)
        output[list_classes] += weight*result[list_classes]
        norm +=weight


    output[list_classes] /= norm

    output.to_csv('output.csv.gz',index=False,compression='gzip')

# post_deal()
# cal_mean()
# post_deal()
bagging()
# get_corr()
# output = pd.read_csv('baseline.csv')
# print(output.isnull().sum())