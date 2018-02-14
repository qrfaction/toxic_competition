import tensorflow     #core dump 需要
import input
from sklearn.cross_validation import KFold
import numpy as np
import tool
import torch
from Ref_Data import  BATCHSIZE
MEAN_TYPE = 'arith_mean'
TFIDF_DIM = 128

LOG = 'log.txt'

def cv(get_model, X, Y, test,K=10,outputfile='baseline.csv.gz'):
    kf = KFold(len(Y), n_folds=K, shuffle=False)

    results = []
    scores = []
    import sys
    with open('log.txt','w') as f:
        sys.output = f
        for i, (train_index, valid_index) in enumerate(kf):
            print('第{}次训练...'.format(i))
            trainset = tool.splitdata( train_index,X)
            label_train = Y[train_index]

            validset = tool.splitdata( valid_index,X)
            label_valid = Y[valid_index]

            model=get_model()
            test_pred,model_score = _train_model(model,test=test,train_x=trainset,train_y=label_train,
                                 val_x=validset,val_y=label_valid)

            scores.append(model_score)
            results.append(test_pred)

    test_predicts = tool.cal_mean(results,scores)
    # test_predicts = tool.cal_mean(results, MEAN_TYPE)


    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    sample_submission = input.read_dataset('sample_submission.csv')
    sample_submission[list_classes] = test_predicts
    sample_submission.to_csv(outputfile, index=False, compression='gzip')

def _train_model(model,train_x, train_y, val_x, val_y,test ,batchsize = BATCHSIZE,frequecy = 100):
    from sklearn.metrics import roc_auc_score

    dataset = tool.CommentData(train_x,train_y)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=2,
    )
    iter = 1
    best_iter = 1
    best_score = -1

    while True:
        for samples_x,samples_f,samples_y in loader:
            model.fit(samples_x,samples_y)
            # evaulate
            if iter % frequecy ==0:
                y_pred = model.predict(val_x)
                Scores = []
                for i in range(6):
                    score = roc_auc_score(val_y[:, i], y_pred[:, i])
                    Scores.append(score)
                mean_score = np.mean(Scores)
                print(mean_score)
                print(Scores)
                if iter == 6*frequecy:
                    best_iter = iter
                    best_score = mean_score
                    weights = Scores
                    test_pred = model.predict(test)      #跑完一次数据集开始预测
                elif best_score < mean_score:
                    best_score = mean_score
                    best_iter = iter
                    if iter > 6*frequecy:             # 分数升高就保存预测结果
                        print(best_score,best_iter,'\n')
                        weights = Scores
                        test_pred = model.predict(test)
                elif iter - best_iter > 5*frequecy:  # patience 为5
                    return test_pred,weights
            iter += 1


def train(maxlen=200):
    wordvecfile = (
                    ('crawl', 300),
                    # ('fasttext',300),
                )
    trainset, testset, labels ,embedding_matrix = \
        input.get_train_test(maxlen,trainfile='clean_train.csv',wordvecfile=wordvecfile)
    embedding_matrix = embedding_matrix['crawl']

    import nnBlock
    getmodel=lambda:nnBlock.DnnModle(300,embedding_matrix,trainable=False,alpha = 3,loss='ceLoss')

    cv(getmodel,trainset,labels,testset,outputfile='baseline.csv.gz',K=6)


if __name__ == "__main__":
    train()