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

    weight = labels.sum()
    weight = labels.shape[0] / weight
    print(weight)



def bagging():
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    PATH = 'results/'
    file_weight = {
        'GRU1.csv.gz':6,
        'GRU2.csv.gz':6,
        'GRU3.csv.gz':6,
        'frGRU1.csv.gz':4,
        'gloveGRU1.csv.gz':4,
        'kernel.csv.gz':26,
        # 'kernel2.csv.gz':10,
        'focalloss.csv.gz':10,
    }
    output = pd.read_csv(PATH + 'GRU1.csv.gz')
    output[list_classes] = 0
    norm = 0
    for file,weight in file_weight.items():
        result = pd.read_csv(PATH+file)
        output[list_classes] += weight*result[list_classes]
        norm +=weight


    output[list_classes] /= norm

    output.to_csv('output.csv.gz',index=False,compression='gzip')


# cal_mean()
# post_deal()
bagging()
