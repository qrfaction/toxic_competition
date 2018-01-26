from keras.layers import Dense, Input
from keras.layers import Conv1D, Embedding
from keras.models import Model
from keras.layers import Bidirectional,  Dropout,GRU,add,Reshape,Multiply,BatchNormalization
from keras.layers.pooling import MaxPool1D
from keras.optimizers import RMSprop
import prepocess
import input
from keras.layers.advanced_activations import PReLU
from keras.layers.merge import concatenate
from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd

TFIDF_DIM = 128



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

    def DenseNet(input_layer, filters):
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
        DBlock4 = concatenate([DBlock4, DBlock3, DBlock2, DBlock1, input_layer])

        return DBlock4

    if name=='res_inception':
        return Res_Inception(input_layer,filters)
    elif name=='DenseNet':
        return DenseNet(input_layer,filters)

class dnn:
    def __init__(self,batch_size,num_words,
                 EMBEDDING_DIM, embedding_matrix, maxlen, trainable=False):

        self.maxlen=maxlen
        self.trainable=trainable
        self.EMBEDDING_DIM=EMBEDDING_DIM
        self.embedding_matrix=embedding_matrix
        self.num_words=num_words

        # tfidf, Input1, Input2, Input3 = self.__tfidfBlock()
        x, sequence_input = self.__commentBlock()

        # combine = concatenate([x,tfidf])

        output = Dense(6, activation="sigmoid")(x)
        X=[sequence_input
            # ,Input1,Input2,Input3
           ]

        self.model = Model(inputs=X, outputs=[output])

        optimizer = RMSprop(clipvalue=1, clipnorm=1)
        self.model.compile(loss='binary_crossentropy',
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

    def __commentBlock(self):
        sequence_input = Input(shape=(self.maxlen,), dtype='int32', name='comment')
        embedding_layer = Embedding(self.num_words,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.maxlen,
                                    trainable=self.trainable)(sequence_input)
        embedding_layer = Dropout(0.3)(embedding_layer)

        layer1 = Conv1D(256,kernel_size=1,padding='same',activation='relu')(embedding_layer)
        attention = Bidirectional(GRU(256,return_sequences=True,activation='sigmoid'),merge_mode='sum')(layer1)
        layer1 = Multiply()([layer1,attention])


        layer2 = Bidirectional(GRU(128,return_sequences=True),merge_mode='sum')(layer1)
        seqlayer = Bidirectional(GRU(64, return_sequences=False),merge_mode='sum')(layer2)

        return seqlayer,sequence_input


    def fit(self,X,Y,verbose=1):
        self.model.fit(X,Y,batch_size=self.batch_size,verbose=verbose,shuffle=False)

    def predict(self,X,verbose=1):
        return self.model.predict(X,verbose=verbose)

    def evaluate(self, X, Y, verbose=1):
        return self.model.evaluate(X, Y, verbose=verbose)

def cv(get_model, X, Y, test,K=10, geo_mean=False):

    def splitdata(index_train,index_valid,dataset):
        train_x={}
        valid_x={}
        for key in dataset.keys():
            train_x[key] = dataset[key][index_train]
            valid_x[key] = dataset[key][index_valid]

        return train_x,valid_x


    kf = KFold(len(Y), n_folds=K, shuffle=False)

    results = []
    total_loss=[]
    for i, (train_index, valid_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        trainset , validset = splitdata( train_index,valid_index ,X)
        label_train = Y[train_index]
        label_valid = Y[valid_index]

        model=get_model()
        model.fit(trainset, label_train)
        loss = model.evaluate(validset, label_valid)  #验证集上的loss、acc

        print("valid set score:", loss)
        total_loss.append(loss)

        results.append(model.predict(test))

    print('total loss',np.array(total_loss).mean(axis=0))

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
    sample_submission.to_csv("baseline.csv.gz", index=False, compression='gzip')

def train(batch_size=256,maxlen=100):

    train, test, labels ,embedding_matrix = input.get_train_test(maxlen)

    getmodel=lambda:dnn(batch_size,len(embedding_matrix),300,embedding_matrix,maxlen=maxlen)

    # model=getmodel()
    # model.fit(train,labels)
    # list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    # sample_submission = input.read_dataset('sample_submission.csv')
    # sample_submission[list_classes] = model.predict(test)
    # sample_submission.to_csv("baseline.csv.gz", index=False, compression='gzip')
    cv(getmodel,train,labels,test)


if __name__ == "__main__":
    train()