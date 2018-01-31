from keras.layers import Dense, Input
from keras.layers import Conv1D, Embedding
from keras.models import Model
from keras.layers import Bidirectional,Dropout,GRU,add,LSTM,Multiply,BatchNormalization
from keras.layers.pooling import MaxPool1D,GlobalAveragePooling1D
from keras.optimizers import RMSprop,Adam
import prepocess
import input
from keras.layers.advanced_activations import PReLU
from keras.layers.merge import concatenate
from sklearn.cross_validation import KFold
import numpy as np
from keras import regularizers
import pandas as pd
from keras import backend as K
import numpy as np
import tensorflow as tf
import tool
import torch.nn.functional as func
import torch.nn as nn
TFIDF_DIM = 128
BATCHSIZE = 256
import torch

def CnnBlock(name,input_layer,filters):
    def Res_Inception(input_layer, filters, activate=True):
        filters = int(filters / 4)
        Ince_5 = Conv1D(filters=filters, kernel_size=1, strides=1, padding='same', activation='relu')(input_layer)
        Ince_5 = Conv1D(filters=filters, kernel_size=3, strides=1, padding='same', activation='relu')(Ince_5)
        Ince_5 = Conv1D(filters=filters, kernel_size=3, strides=1, padding='same')(Ince_5)
        Ince_5 = PReLU()(Ince_5)

        Ince_3 = Conv1D(filters=filters, kernel_size=1, strides=1, padding='same', activation='relu')(input_layer)
        Ince_3 = Conv1D(filters=filters, kernel_size=3, strides=1, padding='same')(Ince_3)
        Ince_3 = PReLU()(Ince_3)

        Ince_pool = MaxPool1D(pool_size=3, strides=1, padding='same')(input_layer)
        Ince_pool = Conv1D(filters=filters, kernel_size=1, strides=1, padding='same')(Ince_pool)
        Ince_pool = PReLU()(Ince_pool)

        Ince_1 = Conv1D(filters=filters, kernel_size=1, strides=1, padding='same')(input_layer)
        Ince_1 = PReLU()(Ince_1)

        Ince = concatenate([Ince_3, Ince_5, Ince_pool, Ince_1], axis=-1)
        if activate == True:
            res_util = add([input_layer, Ince])
            res_util = PReLU()(res_util)
        else:
            res_util = Ince
        return res_util

    def DenseNet(input_layer, filters ):
        filters = int(filters / 4)
        DBlock1 = Conv1D(filters=filters, kernel_size=1, strides=1, padding='same', activation='relu')(input_layer)
        DBlock1 = concatenate([DBlock1, input_layer])
        DBlock1 = Conv1D(filters=filters, kernel_size=1, strides=1, padding='same', activation='relu')(DBlock1)

        DBlock2 = Res_Inception(DBlock1, filters, activate=False)
        DBlock2 = concatenate([DBlock2, DBlock1, input_layer])
        DBlock2 = Conv1D(filters=filters, kernel_size=1, strides=1, padding='same', activation='relu')(DBlock2)

        DBlock3 = Res_Inception(DBlock2, filters, activate=False)
        DBlock3 = concatenate([DBlock3, DBlock2, DBlock1, input_layer])
        DBlock3 = Conv1D(filters=filters, kernel_size=1, strides=1, padding='same', activation='relu')(DBlock3)

        DBlock4 = Res_Inception(DBlock3, filters, activate=False)
        DBlock4 = concatenate([DBlock4, DBlock3, DBlock2, DBlock1])
        DBlock4 = add([DBlock4,input_layer])

        return DBlock4

    if name=='res_inception':
        return Res_Inception(input_layer,filters)
    elif name=='DenseNet':
        return DenseNet(input_layer,filters)


class auc_loss(nn.Module):
    def __init__(self):
        super(auc_loss,self).__init__()
        self.batchsize = 128

    def forword(self,y_true,y_pred):
        y_pred = y_pred/(1-y_pred)

        loss = 0
        for i in range(6):
            yi = y_pred[:,i]
            y_pos = torch.masked_select(yi,y_true.gt(0.5))  # gt greater than
            y_pos = 1/y_pos
            y_neg = torch.masked_select(yi,y_true.lt(0.5))  # lt less than

            loss += torch.sum(torch.mm(y_neg.t(),y_pos))

        return loss/128**2

class dnn:
    def __init__(self,batch_size,
                 EMBEDDING_DIM, embedding_matrix, maxlen, trainable=False):

        self.maxlen=maxlen
        self.trainable=trainable
        self.EMBEDDING_DIM=EMBEDDING_DIM
        self.embedding_matrix=embedding_matrix

        # tfidf, Input1, Input2, Input3 = self.__tfidfBlock()
        x, sequence_input = self.__commentBlock_v1()

        # cF,cF_x  = self.__CountFeature()

        # combine = concatenate([x,cF])

        output = Dense(6, activation="sigmoid")(x)
        X = [
                sequence_input,
            # ,Input1,Input2,Input3
            ]

        self.model = Model(inputs=X, outputs=[output])

        optimizer = RMSprop(lr=0.0015)
        # optimizer = Adam(lr=0.002,clipvalue=1)
        self.model.compile(loss=fbeta_bce_loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])
        self.batch_size=batch_size

    def __tfidfBlock(self):
        Input1 = Input(shape=(TFIDF_DIM,), name='tfidf1')
        Input2 = Input(shape=(TFIDF_DIM,), name='tfidf2')
        Input3 = Input(shape=(TFIDF_DIM,), name='tfidf3')

        tfidf1 = Dense(32, activation='relu')(Input1)

        tfidf2 = Dense(32, activation='relu')(Input2)

        tfidf3 = Dense(32, activation='relu')(Input3)

        tfidf = add([tfidf1,tfidf2,tfidf3])

        return tfidf,Input1,Input2,Input3

    def __commentBlock_v1(self):
        sequence_input = Input(shape=(self.maxlen,), dtype='int32', name='comment')

        output = []
        for file,embedding_matrix in self.embedding_matrix.items() :
            embedding_layer = Embedding(len(embedding_matrix),
                                        self.EMBEDDING_DIM,
                                        weights=[embedding_matrix],
                                        input_length=self.maxlen,
                                        trainable=self.trainable)(sequence_input)
            embedding_layer = Dropout(0.3)(embedding_layer)

            layer2 = Bidirectional(GRU(128,return_sequences=True),merge_mode='sum')(embedding_layer)
            seqlayer = Bidirectional(GRU(64, return_sequences=False),merge_mode='sum')(layer2)
            output.append(seqlayer)
        if len(output) == 1:
            return output[0],sequence_input
        seqlayer = add(output)
        return seqlayer,sequence_input

    def __CountFeature(self):
        x = Input(shape=(18,),name='countFeature')

        fc = Dense(64,activation='relu',kernel_regularizer=regularizers.l1(0.01))(x)

        return fc,x

    def fit(self,X,Y,verbose=1):
        self.model.fit(X,Y,batch_size=self.batch_size,verbose=verbose,shuffle=True)

    def predict(self,X,verbose=1):
        return self.model.predict(X,verbose=verbose)

    def evaluate(self, X, Y, verbose=1):
        return self.model.evaluate(X, Y, verbose=verbose)



def cv(get_model, X, Y, test,K=10, geo_mean=False,outputfile='baseline.csv.gz'):

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

    if geo_mean == True:
        test_predicts = np.ones(results[0].shape)
        for fold_predict in results:
            test_predicts *= fold_predict
        test_predicts **= (1. / len(results))
    else:
        test_predicts = np.zeros(results[0].shape)
        for fold_predict in results:
            test_predicts += fold_predict
        test_predicts /= len(results)


    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    sample_submission = input.read_dataset('sample_submission.csv')
    sample_submission[list_classes] = test_predicts
    sample_submission.to_csv(outputfile, index=False, compression='gzip')

def _train_model(model,train_x, train_y, val_x, val_y,batchsize = 256,frequecy = 100):
    from sklearn.metrics import roc_auc_score

    best_score = -1
    best_iter = 1
    iter = 1

    generator = tool.Generate(train_x,train_y)

    # 从threat 开始
    next_col = 3
    samples_x ,samples_y= generator.genrerate_samples(next_col,batchsize)


    while True:
        model.fit(samples_x,samples_y)

        # evaulate
        if iter % frequecy ==0:
            print("Epoch {0} best_score {1}".format(iter,best_score))
            y_pred = model.predict(val_x)
            Y = val_y
        else :
            y_pred = model.predict(samples_x)
            Y = samples_y

        # 计算下一个需要优化的标签
        Scores = []
        min_score = -1
        for i in range(6):
            score = roc_auc_score(Y,y_pred[:,i])
            Scores.append(score)
            if score < min_score :
                min_score = score
                next_col = i

        mean_score = np.mean(Scores)
        if best_score < mean_score:
            best_score = mean_score
            best_iter = iter
        elif iter - best_iter == 5 :
            break


        samples_x, samples_y = generator.genrerate_samples(next_col, batchsize)
        iter +=1

    return model



def train(batch_size=25600,maxlen=200):
    wordvecfile = (
                    # ('crawl', 300),
                    ('crawl',300),
                )
    trainset, testset, labels ,embedding_matrix = \
        input.get_train_test(maxlen,trainfile='clean_train.csv',wordvecfile=wordvecfile)

    getmodel=lambda:dnn(batch_size,300,embedding_matrix,maxlen=maxlen,trainable=True)

    # train_earlystop(getmodel,trainset,labels,testset)

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