import numpy as np
from tqdm import tqdm
import multiprocessing as mlp
import pandas as pd
PATH='data/'
wordvec={
    'glove42':PATH+'glove.42B.300d.txt',
    'glove840':PATH+'glove.840B.300d.txt',
}

def work(wordmat):
    result={}
    for line in tqdm(wordmat):
        if line is not '':
            wvec = line.split(' ')
            result[wvec[0]] = np.asarray(wvec[1:], dtype='float32')
    return result

def read_wordvec(filename):
    with open(wordvec[filename]) as f:
        wordmat=f.read().split('\n')
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
    print(len(word_dict))
    return word_dict

def read_dataset(filename,cols=None):
    if cols is None:
        return pd.read_csv(PATH+filename)
    return pd.read_csv(PATH+filename,usecols=cols)

def deal_index(filename):
    '处理 to_csv 忘加index=False的问题'
    data=pd.read_csv(PATH+filename)
    data.drop(['Unnamed: 0'],axis=1,inplace=True)
    data.to_csv(PATH+filename,index=False)

if __name__ == "__main__":
    a=read_dataset('labels.csv')
    a.info()
    deal_index('labels.csv')




