import numpy as np
from tqdm import tqdm
import multiprocessing as mlp
import pandas as pd
import prepocess


PATH='data/'
wordvec={
    'glove42':PATH+'glove.42B.300d.txt',
    'glove840':PATH+'glove.840B.300d.txt',
    'crawl':PATH+'crawl-300d-2M.vec',
}

def get_train_test(maxlen,addData=False):
    seqtrain, seqtest = read_dataset('clean_train.csv'), read_dataset('clean_test.csv')



    labels =read_dataset('labels.csv').values


    if addData==True:
        fr,es,de = read_dataset('clean_train_fr.csv'),\
                   read_dataset('clean_train_es.csv'),\
                   read_dataset('clean_train_de.csv'),
        seqtrain = seqtrain.append(fr)
        seqtrain = seqtrain.append(es)
        seqtrain = seqtrain.append(de)
        labels = np.concatenate([labels] * 4 ,axis=0)

    seqtrain, seqtest, embedding_matrix = \
        prepocess.comment_to_seq(seqtrain, seqtest, maxlen=maxlen, wordvecfile='crawl')

    # tfidf_train,tfidf_test=read_dataset('tfidf_train.csv',header=None).values,\
    #                        read_dataset('tfidf_test.csv',header=None).values

    X={
        'comment':seqtrain,
        # 'tfidf1':tfidf_train[:,:128],
        # 'tfidf2': tfidf_train[:,128:256],
        # 'tfidf3': tfidf_train[:,256:],
    }
    testX={
        'comment':seqtest,
        # 'tfidf1':tfidf_test[:,:128],
        # 'tfidf2': tfidf_test[:,128:256],
        # 'tfidf3': tfidf_test[:,256:],
    }
    return X,testX,labels,embedding_matrix

def work(wordmat):
    result={}
    for line in tqdm(wordmat):
        wvec = line.strip(' ').split(' ')
        result[wvec[0]] = np.asarray(wvec[1:], dtype='float32')
    return result

def read_wordvec(filename):
    with open(wordvec[filename]) as f:
        wordmat=f.read().strip('\n').split('\n')
    results = []
    pool = mlp.Pool(mlp.cpu_count())

    aver_t = int(len(wordmat)/mlp.cpu_count()) + 1
    for i in range(mlp.cpu_count()):
        result = pool.apply_async(work, args=( wordmat[i*aver_t:(i+1)*aver_t],))
        results.append(result)
    pool.close()
    pool.join()

    word_dict={}
    for result in results:
        word_dict.update(result.get())
    print('num of words:',len(word_dict))
    return word_dict

def read_dataset(filename,cols=None,header='infer'):
    if cols is None:
        return pd.read_csv(PATH+filename,header=header)
    return pd.read_csv(PATH+filename,usecols=cols,header=header)

def deal_index(filename):
    '处理 to_csv 忘加index=False的问题'
    data=pd.read_csv(PATH+filename)
    data.drop(['Unnamed: 0'],axis=1,inplace=True)
    data.to_csv(PATH+filename,index=False)

if __name__ == "__main__":
    a=read_dataset('labels.csv')
    a.info()
    deal_index('labels.csv')




