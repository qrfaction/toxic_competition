import tensorflow     #core dump 需要
import input
from sklearn.cross_validation import KFold
import numpy as np
import tool
import torch
from Ref_Data import  BATCHSIZE,WEIGHT_FILE
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback


MEAN_TYPE = 'arith_mean'
TFIDF_DIM = 128

LOG = 'log.txt'



class auc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def cv(get_model, X, Y, test,model_name,K=10,outputfile='baseline.csv.gz'):
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
            test_pred,model_score = _train_model(model,model_name+"_"+str(i)+".h5",
                                                 trainset,label_train,
                                                 validset,label_valid,test)

            scores.append(model_score)
            results.append(test_pred)

    test_predicts = tool.cal_mean(results,scores)
    # test_predicts = tool.cal_mean(results, MEAN_TYPE)


    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    sample_submission = input.read_dataset('sample_submission.csv')
    sample_submission[list_classes] = test_predicts
    sample_submission.to_csv(outputfile, index=False, compression='gzip')

def _train_model(model,model_name ,train_x, train_y, val_x, val_y,test ,batchsize = BATCHSIZE,frequecy = 100):
    from sklearn.metrics import roc_auc_score

    generator = tool.Generate(train_x,train_y,batchsize=frequecy*batchsize)

    epoch = 1
    best_epoch = 1
    best_score = -1

    while True:

        samples_x,samples_y = generator.genrerate_samples()
        model.fit(samples_x,samples_y,batch_size=batchsize,epochs=1,verbose=1)

        if epoch >= 20:
            # evaulate
            y_pred = model.predict(val_x, batch_size=2048, verbose=1)
            Scores = []
            for i in range(6):
                score = roc_auc_score(val_y[:, i], y_pred[:, i])
                Scores.append(score)
            cur_score = np.mean(Scores)
            print(cur_score)
            print(Scores)

            if epoch == 20 or best_score < cur_score:
                best_score = cur_score
                best_epoch = epoch
                print(best_score,best_epoch,'\n')
                weights = Scores
                model.save_weights(WEIGHT_FILE + model_name)
            elif epoch - best_epoch > 4 :  # patience 为5
                model.load_weights(WEIGHT_FILE + model_name, by_name=False)
                test_pred = model.predict(test,batch_size=2048)
                return test_pred,weights
        epoch += 1


def train(maxlen=200):
    wordvecfile = (
                    ('crawl', 300),
                    # ('fasttext',300),
                )
    trainset, testset, labels ,embedding_matrix = \
        input.get_train_test(maxlen,trainfile='clean_train.csv',wordvecfile=wordvecfile)
    embedding_matrix = embedding_matrix['crawl']

    import nnBlock
    getmodel=lambda:nnBlock.get_model(embedding_matrix,trainable=False)

    cv(getmodel,trainset,labels,testset,outputfile='baseline.csv.gz',K=5,model_name="GRU1")


if __name__ == "__main__":
    train()