import tensorflow     #core dump 需要
import input
from sklearn.cross_validation import KFold
import numpy as np
import tool
from Ref_Data import  BATCHSIZE,WEIGHT_FILE,USE_CHAR_VEC


def cv(model_para, X, Y, test,K=10,outputfile='baseline.csv.gz'):
    import nnBlock
    def get_model(model_para):
        m =  nnBlock.model(
            embedding_matrix=model_para['embedding_matrix'],
            trainable=model_para['trainable'],
            load_weight =model_para['load_weight'],
            loss=model_para['loss'],
            char_weight = model_para['char_weight']
        )
        m.get_layer(model_para['modelname'])
        return m

    kf = KFold(len(Y), n_folds=K, shuffle=False)

    results = []
    scores = []

    for i, (train_index, valid_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        trainset = tool.splitdata( train_index,X)
        label_train = Y[train_index]

        validset = tool.splitdata( valid_index,X)
        label_valid = Y[valid_index]

        model=get_model(model_para)
        test_pred,model_score = _train_model(model,
                                             model_para['modelname']+"_"+str(i)+".h5",
                                             trainset,label_train,
                                             validset,label_valid,test)

        scores.append(model_score)
        results.append(test_pred)

    test_predicts = tool.cal_mean(results,scores)

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

        if epoch >= 10:
            # evaulate
            y_pred = model.predict(val_x, batch_size=2048, verbose=1)
            Scores = []
            for i in range(6):
                score = roc_auc_score(val_y[:, i], y_pred[:, i])
                Scores.append(score)
            cur_score = np.mean(Scores)
            print(cur_score)
            print(Scores)

            if epoch == 10 or best_score < cur_score:
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
    if USE_CHAR_VEC:
        trainset, testset, labels, embedding_matrix,char_weight = \
            input.get_train_test(maxlen, trainfile='clean_train.csv', wordvecfile=wordvecfile)
        embedding_matrix = embedding_matrix['crawl']
    else:
        trainset, testset, labels ,embedding_matrix = \
        input.get_train_test(maxlen,trainfile='clean_train.csv',wordvecfile=wordvecfile)
    embedding_matrix = embedding_matrix['crawl']

    """
        loss :  focalLoss   diceLoss  binary_crossentropy
        modelname : rnn cnn cnnGLU
    """

    model_para = {
        'embedding_matrix':embedding_matrix,
        'trainable':False,
        'loss':'focalLoss',
        'load_weight' :False,
        'modelname':'rnn',
        'char_weight':None
    }
    if USE_CHAR_VEC:
        model_para['char_weight'] = char_weight

    cv(model_para,trainset,labels,testset,outputfile='baseline.csv.gz',K=5)


if __name__ == "__main__":
    train()