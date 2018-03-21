import input
from sklearn.cross_validation import KFold
import numpy as np
import tool
from Ref_Data import  BATCHSIZE,WEIGHT_FILE,USE_CHAR_VEC




def save_result(test_predicts,outputfile):
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    sample_submission = input.read_dataset('sample_submission.csv')
    sample_submission[list_classes] = test_predicts
    sample_submission.to_csv(outputfile, index=False, compression='gzip')

def cv(model_para, X, Y, test,K=10,outputfile='baseline.csv.gz',balance_sample=False):
    import nnBlock
    def get_model(model_para):
        m =  nnBlock.model(
            embedding_matrix=model_para['embedding_matrix'],
            trainable=model_para['trainable'],
            load_weight =model_para['load_weight'],
            loss=model_para['loss'],
            char_weight = model_para['char_weight'],
            setting= model_para['setting'],
            maxlen=model_para['maxlen'],
        )
        m.get_layer(model_para['modelname'])
        return m

    kf = KFold(len(Y), n_folds=K, shuffle=False)

    results = []
    scores = []


    fit_train = _train_model

    for i, (train_index, valid_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        trainset = tool.splitdata( train_index,X)
        label_train = Y[train_index]

        validset = tool.splitdata( valid_index,X)
        label_valid = Y[valid_index]

        model=get_model(model_para)
        test_pred,model_score = fit_train(model,
                                             model_para['modelname']+"_"+str(i)+".h5",
                                             trainset,label_train,
                                             validset,label_valid,test,window_size=model_para['window_size'])

        scores.append(model_score)
        results.append(test_pred)

    test_predicts = tool.cal_mean(results,scores)
    save_result(test_predicts,outputfile)


def _train_model(model,model_name ,train_x, train_y, val_x, val_y,test ,batchsize = BATCHSIZE,frequecy = 50,window_size=0):
    from sklearn.metrics import roc_auc_score

    generator = tool.Generate(train_x,train_y,batchsize=frequecy*batchsize,window_size=window_size)

    epoch = 1
    best_epoch = 1
    best_score = -1000

    while True:

        samples_x,samples_y = generator.genrerate_samples()
        model.fit(samples_x,samples_y,batch_size=batchsize,epochs=1,verbose=1)


        # if epoch == 15:
        #     model.recompile(loss='rankLoss')


        if epoch >= 10:
            # evaulate
            y_pred = model.predict(val_x, batch_size=2048, verbose=1)
            Scores = []
            for i in range(6):
                score = roc_auc_score(val_y[:, i], y_pred[:, i])
                Scores.append(score)
            print(Scores)

            cur_score = np.mean(Scores)
            print(cur_score)

            if  best_score < cur_score:
                best_score = cur_score
                best_epoch = epoch
                print(best_score,best_epoch,'\n')
                weights = Scores
                model.save_weights(WEIGHT_FILE + model_name)
            elif epoch - best_epoch > 12 :  # patience 为10
                model.load_weights(WEIGHT_FILE + model_name, by_name=False)
                test_pred = model.predict(test,batch_size=2048)
                return test_pred,weights
        epoch += 1


def train(maxlen=200,outputfile='baseline.csv.gz',wordvec='crawl'):
    if wordvec=='crawl':
        wordvecfile = (
                        (wordvec, 300),
                        # ('fasttext',300),
                    )
    elif wordvec=='glove':
        wordvecfile = (
            (wordvec, 200),
            # ('fasttext',300),
        )
    if USE_CHAR_VEC:
        trainset, testset, labels, embedding_matrix,char_weight = \
            input.get_train_test(maxlen, trainfile='clean_train.csv', wordvecfile=wordvecfile)
        embedding_matrix = embedding_matrix[wordvec]
    else:
        trainset, testset, labels ,embedding_matrix = \
        input.get_train_test(maxlen,trainfile='clean_train.csv',wordvecfile=wordvecfile)
    embedding_matrix = embedding_matrix[wordvec]

    """
        loss :  focalLoss     binary_crossentropy  rankLoss
        modelname : rnn cnn 
    """

    model_para = {
        'embedding_matrix': embedding_matrix,
        'trainable': False,
        'loss': 'focalLoss',
        'load_weight': False,
        'modelname': 'rnn',
        'char_weight': None,
        'maxlen':maxlen,
        'window_size': 20,
        'setting':{
            'lr':0.0008,
            'decay':0.004,
            'dropout': 0.3,
            'size1':170,    #rnn
            'size2':80,
            # 'size1':256,
            # 'size2':128,
        }
    }

    # model_para['setting']['size1'] = 180
    cv(model_para, trainset, labels, testset, outputfile='rnnflLoss1.csv.gz', K=10)

    model_para['loss'] = 'binary_crossentropy'
    cv(model_para, trainset, labels, testset, outputfile='rnnceLoss1.csv.gz', K=10)



    # model_para['setting']['size1'] = 190
    # cv(model_para, trainset, labels, testset, outputfile='rnnfl1Loss2.csv.gz', K=10)

    # model_para['setting']['size1'] = 200
    # cv(model_para, trainset, labels, testset, outputfile='rnnflLoss3.csv.gz', K=10)
    #
    # model_para['setting']['size1'] = 160
    # cv(model_para, trainset, labels, testset, outputfile='rnnflLoss4.csv.gz', K=10)


    # model_para['setting']['size1'] = 190
    # model_para['setting']['size2'] = 80
    # cv(model_para, trainset, labels, testset, outputfile='rnnfocalLoss5.csv.gz', K=10)
    #
    # model_para['setting']['size1'] = 160
    # model_para['setting']['size2'] = 90
    # cv(model_para, trainset, labels, testset, outputfile='rnnfocalLoss6.csv.gz', K=10)
    #
    # model_para['setting']['size1'] = 170
    # model_para['setting']['size2'] = 90
    # cv(model_para, trainset, labels, testset, outputfile='rnnfocalLoss7.csv.gz', K=10)



if __name__ == "__main__":
    train(200)