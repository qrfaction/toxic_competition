import pandas as pd
from scipy.stats import ks_2samp


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


def corr(first_file, second_file):
    result = 'results/'
    # assuming first column is `class_name_id`
    first_df = pd.read_csv(result+first_file, index_col=0)
    second_df = pd.read_csv(result+second_file, index_col=0)
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    for class_name in class_names:
        # all correlations
        print('\n Class: %s' % class_name)
        print(' Pearson\'s correlation score: %0.6f' %
              first_df[class_name].corr(
                  second_df[class_name], method='pearson'))
        print(' Kendall\'s correlation score: %0.6f' %
              first_df[class_name].corr(
                  second_df[class_name], method='kendall'))
        print(' Spearman\'s correlation score: %0.6f' %
              first_df[class_name].corr(
                  second_df[class_name], method='spearman'))
        ks_stat, p_value = ks_2samp(first_df[class_name].values,
                                    second_df[class_name].values)
        print(' Kolmogorov-Smirnov test:    KS-stat = %.6f    p-value = %.3e\n'
              % (ks_stat, p_value))

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
        # "celoss64.csv.gz":5,
        # "cnnensemble.csv.gz":1,
        # "rnnensemble.csv.gz":18,
        # "rnnflensemble.csv.gz":20,
        # "62_63To65.csv.gz":2,
        "l1.csv.gz":3,
        "l2.csv.gz":3,
        # "ensemble.csv.gz":10,/
        "blend_it_all.csv.gz":10,

    }
    # file_weight = dict([('rnnfl1Loss'+str(i)+'.csv.gz',1) for i in range(1,5)])
    print(file_weight)
    output = pd.read_csv(PATH+'l1.csv.gz')
    output[list_classes] = 0
    norm = 0
    for file,weight in file_weight.items():
        result = pd.read_csv(PATH+file)
        output[list_classes] += weight*result[list_classes]
        norm +=weight


    output[list_classes] /= norm

    output.to_csv('ensemble.csv.gz',index=False,compression='gzip')




# post_deal()
# cal_mean()
# post_deal()
bagging()
# corr('rnnensemble.csv.gz',"62_63To65.csv.gz")
# output = pd.read_csv('baseline.csv')
# print(output.isnull().sum())