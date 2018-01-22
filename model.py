from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, Embedding
from keras.models import Model
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout,GRU,add
from keras.layers.pooling import MaxPool1D
from keras.optimizers import Adam
import prepocess
import input
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU
from keras.layers.merge import concatenate
from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd
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
        sequence_input = Input(shape=(maxlen,), dtype='int32')
        embedding_layer = Embedding(num_words,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=maxlen,
                                    trainable=trainable)(sequence_input)
        x = Bidirectional(GRU(64, return_sequences=True))(embedding_layer)
        x = Dropout(0.3)(x)
        x = Bidirectional(GRU(64, return_sequences=False))(x)
        x = Dense(32, activation="relu")(x)

        output = Dense(6, activation="sigmoid")(x)
        self.model = Model(inputs=[sequence_input], outputs=[output])
        optimizer = Adam(lr=0.001)
        self.model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        self.batch_size=batch_size

    def fit(self,X,Y,verbose=1,initial_epoch=0):
        self.model.fit(X,Y,batch_size=self.batch_size,verbose=verbose,initial_epoch=initial_epoch)

    def predict(self,X,verbose=1):
        y=self.model.predict(X,verbose=verbose)
        return y


def cv(get_model, X, Y, test, K=10, seed=2018, geo_mean=False):
    kf = KFold(len(X), n_folds=K, shuffle=True, random_state=seed)

    results = []
    for i, (train_index, valid_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        trainset = X[train_index]
        label_train = Y[train_index]

        model=get_model()
        model.fit(trainset, label_train)
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
    sample_submission.to_csv("baseline.csv.gz", index=False, compression='gzip')

def train(maxlen=100):


    train,test=input.read_dataset('clean_train.csv'),input.read_dataset('clean_test.csv')
    labels=input.read_dataset('labels.csv').values
    train, test,embedding_matrix=prepocess.comment_to_seq(train,test,maxlen=maxlen)

    getmodel=lambda:dnn(128,len(embedding_matrix),300,embedding_matrix,maxlen=maxlen)

    cv(getmodel,train,labels,test)


if __name__ == "__main__":
    train()