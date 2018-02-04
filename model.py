import nnBlock
import input
from sklearn.cross_validation import KFold
import numpy as np
import tool
import torch


MEAN_TYPE = 'arith_mean'
TFIDF_DIM = 128
BATCHSIZE = 256


def cv(get_model, X, Y, test,K=10,outputfile='baseline.csv.gz'):

    kf = KFold(len(Y), n_folds=K, shuffle=False)

    results = []
    for i, (train_index, valid_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        trainset = tool.splitdata( train_index,X)
        label_train = Y[train_index]

        validset = tool.splitdata( valid_index,X)
        label_valid = Y[valid_index]

        model=get_model()
        model = _train_model(model,train_x=trainset,train_y=label_train,
                             val_x=validset,val_y=label_valid)
        results.append(model.predict(test))

    test_predicts = tool.cal_mean(results,MEAN_TYPE)


    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    sample_submission = input.read_dataset('sample_submission.csv')
    sample_submission[list_classes] = test_predicts
    sample_submission.to_csv(outputfile, index=False, compression='gzip')

def _train_model(model,train_x, train_y, val_x, val_y,batchsize = 256,frequecy = 100):
    from sklearn.metrics import roc_auc_score

    dataset = tool.CommentData(train_x,train_y)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=batchsize,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多线程来读数据
    )
    iter = 1
    best_score = -1
    for epoch in range(1000000000):
        for samples_x,samples_y in loader:
            model.fit(samples_x,samples_y,'celoss')
            # evaulate
            if iter % frequecy ==0:
                y_pred = model.predict(val_x)
                Scores = []
                for i in range(6):
                    score = roc_auc_score(val_y[:, i], y_pred[:, i])
                    Scores.append(score)
                mean_score = np.mean(Scores)
                print("iters {0} best_score {1}".format(iter, best_score))
                print(mean_score)
                print(Scores)
                if best_score < mean_score:
                    best_score = mean_score
                else:
                    break
            iter +=1

    return model


def train(maxlen=200):
    wordvecfile = (
                    # ('crawl', 300),
                    ('crawl',300),
                )
    trainset, testset, labels ,embedding_matrix = \
        input.get_train_test(maxlen,trainfile='clean_train.csv',wordvecfile=wordvecfile)
    embedding_matrix = embedding_matrix['crawl']

    getmodel=lambda:nnBlock.DnnModle(300,embedding_matrix,trainable=True,alpha = 2)

    # model = getmodel()
    #
    # model.fit(trainset,labels)
    #
    # list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    # sample_submission = input.read_dataset('sample_submission.csv')
    # sample_submission[list_classes] = model.predict(testset)
    # sample_submission.to_csv("baseline.csv.gz", index=False, compression='gzip')


    cv(getmodel,trainset,labels,testset,outputfile='baseline.csv.gz',K=6)


if __name__ == "__main__":
    train()