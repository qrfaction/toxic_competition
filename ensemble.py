import nnBlock
from sklearn.cross_validation import KFold
import tool
import numpy as np
from Ref_Data import  BATCHSIZE,WEIGHT_FILE,USE_CHAR_VEC
import input

def get_weight(y_scores,y,old_weight):
    y_pred = np.around(y)
    error = (y_pred!=y)
    error = error.mean()
    error = 0.5*np.log((1-error)/error)
    y = (y-0.5)*2
    y_scores = (y_scores-0.5)*2
    weight = np.exp(-error*y_scores*y)
    weight = weight.mean(axis=-1)
    weight = old_weight*weight
    Z = len(y)/weight.sum()
    weight*=Z
    return weight

def Boost(X,Y,test,para,setting,outputfile='boost.csv.gz',cal_weight = False):

    def _train_model(model, model_name,train_x, train_y,val_x, val_y,
                     test,batchsize=BATCHSIZE, frequecy=50,init=30):
        from sklearn.metrics import roc_auc_score

        generator = tool.Generate(train_x, train_y, batchsize=frequecy * batchsize)

        epoch = 1
        best_epoch = 1
        best_score = -1

        while True:

            samples_x, samples_y = generator.genrerate_samples()
            model.fit(samples_x, samples_y, batch_size=batchsize, epochs=1, verbose=0)

            if epoch >= init:
                # evaulate
                y_pred = model.predict(val_x, batch_size=2048, verbose=0)
                Scores = []
                for i in range(6):
                    score = roc_auc_score(val_y[:, i], y_pred[:, i])
                    Scores.append(score)
                cur_score = np.mean(Scores)
                print(cur_score)
                print(Scores)

                if epoch == init or best_score < cur_score:
                    best_score = cur_score
                    best_epoch = epoch
                    print(best_score, best_epoch, '\n')
                    result = y_pred
                    model.save_weights(WEIGHT_FILE + model_name)
                elif epoch - best_epoch > 12:  # patience 为5
                    model.load_weights(WEIGHT_FILE + model_name, by_name=False)
                    test_pred = model.predict(test, batch_size=2048)
                    return test_pred, result
            epoch += 1

    def get_model(model_para):
        m = nnBlock.model(
            embedding_matrix=model_para['embedding_matrix'],
            trainable=model_para['trainable'],
            load_weight=model_para['load_weight'],
            loss=model_para['loss'],
            boost=model_para['boost'],
        )
        m.get_layer(model_para['modelname'])
        return m

    def cv(model_para, X, Y, test, K=5,init=30,sample_weight=None):


        kf = KFold(len(Y), n_folds=K, shuffle=False)

        results = []
        train_score = np.zeros((len(Y),6))

        for i, (train_index, valid_index) in enumerate(kf):
            print('第{}次训练...'.format(i))
            trainset = tool.splitdata(train_index, X)
            label_train = Y[train_index]

            validset = tool.splitdata(valid_index, X)
            label_valid = Y[valid_index]

            model = get_model(model_para)
            test_pred, val_score = _train_model(model,
                                model_para['modelname'] + "_" + str(i) + ".h5",
                                trainset, label_train,
                                validset, label_valid, test,init=init)

            train_score[valid_index] = val_score
            results.append(test_pred)

        test_predicts = tool.cal_mean(results, None)
        return train_score,test_predicts

    train_score,test_score = cv(para,X,Y,test,init=30)
    para['boost'] = True
    if cal_weight:
        para['sample_weight'] = np.ones(len(Y))
    for loss,model_name in setting:
        X['boost'] = train_score
        test['boost'] = test_score
        para['loss'] = loss
        para['modelname'] = model_name
        if cal_weight:
            para['sample_weight'] = get_weight(train_score,Y,para['sample_weight'])
        train_score,test_score = cv(para, X, Y, test,init=5)


    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    sample_submission = input.read_dataset('sample_submission.csv')
    sample_submission[list_classes] = test_score
    sample_submission.to_csv(outputfile, index=False, compression='gzip')

def ensemble_boost(maxlen=200,wordvec='crawl'):

    wordvecfile = (
        (wordvec, 300),
    )

    trainset, testset, labels, embedding_matrix = \
            input.get_train_test(maxlen, trainfile='clean_train.csv', wordvecfile=wordvecfile)
    embedding_matrix = embedding_matrix[wordvec]

    """
        loss :  focalLoss   diceLoss  binary_crossentropy
        modelname : rnn cnn cnnGLU
    """

    model_para = {
        'embedding_matrix': embedding_matrix,
        'trainable': False,
        'loss': 'focalLoss',
        'load_weight': False,
        'modelname': 'rnn',
        'char_weight': None,
        'boost':False,
        'sample_weight':None,
    }
    setting = [
        ['focalLoss','rnn'],
        ['focalLoss','rnn']
    ]
    Boost(trainset,labels,testset,model_para,setting,outputfile='focal.csv.gz')

    setting = [
        ['focalLoss', 'rnn'],
        ['focalLoss', 'rnn']
    ]
    Boost(trainset, labels, testset, model_para, setting,cal_weight=True,outputfile='focal_cal_weight.csv.gz')

    setting = [
        ['focalLoss', 'rnn'],
        ['binary_crossentropy', 'rnn']
    ]
    Boost(trainset, labels, testset, model_para, setting, cal_weight=True, outputfile='cal_weight.csv.gz')

    from train import train
    train()


import pandas as pd
import numpy as np
import os

# Controls weights when combining predictions
# 0: equal average of all inputs;
# 1: up to 50% of weight going to least correlated input
DENSITY_COEFF = 0.1
assert DENSITY_COEFF >= 0.0 and DENSITY_COEFF <= 1.0

# When merging 2 files with corr > OVER_CORR_CUTOFF
# the result's density is the max instead of the sum of the merged files' densities
OVER_CORR_CUTOFF = 0.98
assert OVER_CORR_CUTOFF >= 0.0 and OVER_CORR_CUTOFF <= 1.0

INPUT_DIR = '../input/private-toxic-comment-sumbmissions/'

def load_submissions():
    files = os.listdir(INPUT_DIR)
    csv_files = []
    for f in files:
        if f.endswith(".csv"):
            csv_files.append(f)
    frames = {f:pd.read_csv(INPUT_DIR+f).sort_values('id') for f in csv_files}
    return frames


def get_corr_mat(col,frames):
    c = pd.DataFrame()
    for name,df in frames.items():
        c[name] = df[col]
    cor = c.corr()
    for name in cor.columns:
        cor.set_value(name,name,0.0)
    return cor


def highest_corr(mat):
    n_cor = np.array(mat.values)
    corr = np.max(n_cor)
    idx = np.unravel_index(np.argmax(n_cor, axis=None), n_cor.shape)
    f1 = mat.columns[idx[0]]
    f2 = mat.columns[idx[1]]
    return corr,f1,f2


def get_merge_weights(m1,m2,densities):
    d1 = densities[m1]
    d2 = densities[m2]
    d_tot = d1 + d2
    weights1 = 0.5*DENSITY_COEFF + (d1/d_tot)*(1-DENSITY_COEFF)
    weights2 = 0.5*DENSITY_COEFF + (d2/d_tot)*(1-DENSITY_COEFF)
    return weights1, weights2


def ensemble_col(col,frames,densities):
    if len(frames) == 1:
        _, fr = frames.popitem()
        return fr[col]

    mat = get_corr_mat(col,frames)
    corr,merge1,merge2 = highest_corr(mat)
    new_col_name = merge1 + '_' + merge2

    w1,w2 = get_merge_weights(merge1,merge2,densities)
    new_df = pd.DataFrame()
    new_df[col] = (frames[merge1][col]*w1) + (frames[merge2][col]*w2)
    del frames[merge1]
    del frames[merge2]
    frames[new_col_name] = new_df

    if corr >= OVER_CORR_CUTOFF:
        print('\t',merge1,merge2,'  (OVER CORR)')
        densities[new_col_name] = max(densities[merge1],densities[merge2])
    else:
        print('\t',merge1,merge2)
        densities[new_col_name] = densities[merge1] + densities[merge2]

    del densities[merge1]
    del densities[merge2]
    #print(densities)
    return ensemble_col(col,frames,densities)


ens_submission = pd.read_csv('results/sample_submission.csv').sort_values('id')
#print(get_corr_mat('toxic',load_submissions()))

for col in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
    frames = load_submissions()
    print('\n\n',col)
    densities = {k:1.0 for k in frames.keys()}
    ens_submission[col] = ensemble_col(col,frames,densities)

print(ens_submission)
ens_submission.to_csv('lazy_ensemble_submission.csv', index=False)


if __name__ == "__main__":
    ensemble_boost()