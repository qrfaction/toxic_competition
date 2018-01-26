import numpy as np
from tqdm import tqdm
import pandas as pd
import embedding

PATH='data/'
wordvec={
    'glove42':PATH+'glove.42B.300d.txt',
    'glove840':PATH+'glove.840B.300d.txt',
    'crawl':PATH+'crawl-300d-2M.vec',
}
UNKONW='unknow'

usecols = [
    ##count feature
    'total_length',
    'capitals',
    'caps_vs_length',
    'num_exclamation_marks',
    'num_question_marks',
    'num_punctuation',
    'num_symbols',
    'num_words',
    'num_unique_words',
    'words_vs_unique',
    'num_smilies',
    'count_word',
    'count_unique_word',
    "count_punctuations",
    "count_stopwords",
    "mean_word_len",
    'word_unique_percent',
    'punct_percent',

    ## leaky feature
    # 'ip',
    # 'count_ip',
    # 'link',
    # 'count_links',
    # 'article_id',
    # 'article_id_flag',
    # 'username',
    # 'count_usernames',
]


def get_train_test(maxlen,addData=False,wordvecfile='crawl',dimension=300):
    train, test = read_dataset('clean_train.csv',cols=usecols), read_dataset('clean_test.csv',cols=usecols)

    labels =read_dataset('labels.csv').values

    if addData==True:
        fr,es,de = read_dataset('clean_train_fr.csv'),\
                   read_dataset('clean_train_es.csv'),\
                   read_dataset('clean_train_de.csv'),
        seqtrain = seqtrain.append(fr)
        seqtrain = seqtrain.append(es)
        seqtrain = seqtrain.append(de)
        labels = np.concatenate([labels] * 4 ,axis=0)

    train['comment_text'].fillna(UNKONW, inplace=True)
    test['comment_text'].fillna(UNKONW, inplace=True)
    text = train['comment_text'].values.tolist() + test['comment_text'].values.tolist()

    sequences, embedding_matrix = embedding.get_embedding_matrix(text, maxlen, dimension, wordvecfile)
    trainseq = sequences[:len(train)]
    testseq = sequences[len(train):]
    assert len(trainseq) == len(train)
    assert len(testseq) == len(test)

    X={
        'comment':trainseq,
        'countFeature':train[usecols],
        # 'tfidf1':tfidf_train[:,:128],
        # 'tfidf2': tfidf_train[:,128:256],
        # 'tfidf3': tfidf_train[:,256:],
    }
    testX={
        'comment':testseq,
        'countFeature':test[usecols],
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

    import multiprocessing as mlp

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




