import nnBlock
from sklearn.cross_validation import KFold
import tool
import numpy as np
from Ref_Data import  BATCHSIZE,WEIGHT_FILE,USE_CHAR_VEC
import input

def Boost(X,Y,test,para,num_model = 3,outputfile='boost.csv.gz'):

    def _train_model(model, model_name,train_x, train_y,val_x, val_y,test,batchsize=BATCHSIZE, frequecy=50,init=30):
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
            boost=model_para['boost']
        )
        m.get_layer(model_para['modelname'])
        return m

    def cv(model_para, X, Y, test, K=5,init=30):


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
    for i in range(num_model):
        X['boost'] = train_score
        test['boost'] = test_score
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
        'boost':False
    }
    Boost(trainset,labels,testset,model_para)

if __name__ == "__main__":
    ensemble_boost()