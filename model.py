from keras.layers import Dense, Input
from keras.layers import Conv1D, Embedding
from keras.models import Model
from keras.layers import Bidirectional,Dropout,GRU,add,LSTM,Multiply,BatchNormalization
from keras.optimizers import RMSprop,Adam
import nnBlock
import input
from sklearn.cross_validation import KFold
import numpy as np
from keras import regularizers
import numpy as np
import tool


TFIDF_DIM = 128
BATCHSIZE = 256




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

    generator = tool.Generate(train_x,train_y,batchsize)

    # 从threat 开始
    while True:
        samples_x, samples_y = generator.genrerate_samples()
        model.fit(samples_x,samples_y)
        # evaulate
        if iter % frequecy ==0:
            print("Epoch {0} best_score {1}".format(iter,best_score))
            y_pred = model.predict(val_x)
            Scores = []
            for i in range(6):
                score = roc_auc_score(val_y[:, i], y_pred[:, i])
                Scores.append(score)
            mean_score = np.mean(Scores)
            if best_score < mean_score:
                best_score = mean_score
                best_iter = iter
            elif iter - best_iter >= 2*frequecy:
                break

        iter +=1

    return model

def train(batch_size=256,maxlen=200):
    wordvecfile = (
                    # ('crawl', 300),
                    ('crawl',300),
                )
    trainset, testset, labels ,embedding_matrix = \
        input.get_train_test(maxlen,trainfile='clean_train.csv',wordvecfile=wordvecfile)
    embedding_matrix = embedding_matrix['crawl']
    # getmodel=lambda:dnn(batch_size,300,embedding_matrix,maxlen=maxlen,trainable=True)
    getmodel=lambda:nnBlock.DnnModle(batch_size,300,embedding_matrix,trainable=True)
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