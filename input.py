import embedding
import numpy as np
from tqdm import tqdm
import pandas as pd
from Ref_Data import replace_word,PATH,USE_POSTAG,USE_CHAR_VEC,WEIGHT_FILE,LEN_CHAR_SEQ,USE_TFIDF,CHAR_N

wordvec={
    'glove42':PATH+'glove.42B.300d.txt',
    'glove840':PATH+'glove.840B.300d.txt',
    'crawl':PATH+'crawl-300d-2M.vec',
    'word2vec':PATH+'word2vec.txt',
    'fasttext':PATH+'wiki.en.bin',
    'glove':PATH+'glove.twitter.27B.200d.txt',
}

# 使用哪些特征
usecols = [
    'comment_text',
    ##count feature



    'count_sent',
    # 'total_length',
    # 'count_unique_word',
    'capitals',
    "mean_word_len",
    # 'caps_vs_length',


    # 'toxicity_score_level',
    'quoting_attacklevel',
    # 'recipient_attack_level',
    'third_party_attacklevel',
    'other_attacklevel',
    # 'toxicity_level',
    'attacklevel',

]

from Ref_Data import NUM_TOPIC,USE_LETTERS,USE_TOPIC

if USE_TOPIC:
    usecols += ['topic' + str(i) for i in range(NUM_TOPIC)]
if USE_LETTERS:
    usecols += ['distri_'+chr(i) for i in range(97,26+97)]
    usecols.append('distri_!')
if USE_CHAR_VEC:
    usecols.append('char_text')
if USE_TFIDF:
    usecols += ["tfidf" + str(x) for x in range(CHAR_N)]



def get_train_test(maxlen,trainfile='clean_train.csv',wordvecfile=(('fasttext',300),)):
    """
    :param maxlen: 句子最大长度
    :param trainfile: 使用哪个训练集版本(有多种语言翻译后的版本)
    :param wordvecfile: 使用哪个词向量
    :return: 清洗完的训练集 ,测试集 ,标签 ,词向量矩阵
    """

    train, test = read_dataset(trainfile,cols=usecols), read_dataset('clean_test.csv',cols=usecols)

    labels =read_dataset('labels.csv').values

    train['comment_text'].fillna(replace_word['unknow'], inplace=True)
    test['comment_text'].fillna(replace_word['unknow'], inplace=True)
    text = train['comment_text'].values.tolist() + test['comment_text'].values.tolist()


    print('tokenize word')
    print(wordvecfile[0])


    sentences, word_index, frequency = embedding.tokenize_sentences(text)

    from keras.preprocessing.sequence import pad_sequences
    sentences = pad_sequences(sentences, maxlen=maxlen, truncating='post')
    sequences = np.array(sentences)

    trainseq = sequences[:len(train)]
    testseq = sequences[len(train):]

    def normlizer(train,test):
        normilze_feature = [
            # 'toxicity_score_level',
            'quoting_attacklevel',
            # 'recipient_attack_level',
            'third_party_attacklevel',
            'other_attacklevel',
            # 'toxicity_level',
            'attacklevel',

            'count_sent',
            # 'total_length',
            # 'count_unique_word',
            'capitals',
            "mean_word_len",
            # 'caps_vs_length',
        ]

        dataset = train.append(test)

        print(dataset[normilze_feature].describe())
        for col in normilze_feature:
            train[col] = (train[col] - dataset[col].mean()) / dataset[col].std()
            test[col] = (test[col] - dataset[col].mean()) / dataset[col].std()

        return train,test

    train,test = normlizer(train,test)

    usecols.remove('comment_text')
    if USE_CHAR_VEC:
        usecols.remove('char_text')
    X={
        'comment':trainseq,
        'countFeature':train[usecols].values,
        # 'tfidf1':tfidf_train[:,:128],
        # 'tfidf2': tfidf_train[:,128:256],
        # 'tfidf3': tfidf_train[:,256:],
    }
    testX={
        'comment':testseq,
        'countFeature':test[usecols].values,
        # 'tfidf1':tfidf_test[:,:128],
        # 'tfidf2': tfidf_test[:,128:256],
        # 'tfidf3': tfidf_test[:,256:],
    }

    embedding_matrix = embedding.get_wordvec(word_index, frequency, wordvecfile)
    if USE_CHAR_VEC:
        char_text = train['char_text'].tolist()
        char_text += test['char_text'].tolist()
        char_seq = embedding.get_char_num_seq(char_text)
        char_seq = pad_sequences(char_seq, maxlen=LEN_CHAR_SEQ, truncating='post')
        train_ch = char_seq[:len(train)]
        test_ch = char_seq[len(train):]
        X['char'] = train_ch
        testX['char'] = test_ch
        char_weight = np.load(WEIGHT_FILE+"char_vec.npy")
        return X,testX,labels,embedding_matrix,char_weight

    return X,testX,labels,embedding_matrix

def get_transfer_data(maxlen,fastText,trainfile,target):

    train = read_dataset(trainfile)
    # labels = read_dataset(tar).values
    labels = train[target].values

    train['comment_text'].fillna(replace_word['unknow'], inplace=True)
    text = train['comment_text'].values.tolist()

    from keras.preprocessing.sequence import pad_sequences
    print('tokenize word')


    sentences, word_index, frequency = embedding.tokenize_sentences(text)

    sentences = pad_sequences(sentences, maxlen=maxlen, truncating='post')
    sequences = np.array(sentences)

    trainseq = sequences
    assert len(trainseq) == len(train)

    usecols.remove('comment_text')

    X = {
        'comment': trainseq,
    }


    def get_embedding_matrix(word_index,fastText):
        from fastText import load_model
        import input
        print('get embedding matrix')

        num_words = len(word_index) + 1
        # 停止符用0
        embedding_matrix = np.random.uniform(-0.2,0.2,(num_words, 300))
        print(embedding_matrix.shape)


        # ft_model = load_model(PATH + fastText)
        embeddings_index = input.read_wordvec(fastText)
        # for word, i in tqdm(word_index.items()):
        #     embedding_matrix[i] = ft_model.get_word_vector(word).astype('float32')
        for word, i in tqdm(word_index.items()):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                continue
            else:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    embedding_matrix = get_embedding_matrix(word_index,fastText)

    return X, labels, embedding_matrix

def work(wordmat):
    result={}
    for line in tqdm(wordmat):
        wvec = line.strip('\n').strip(' ').split(' ')
        result[wvec[0]] = np.asarray(wvec[1:], dtype='float32')

    return result

def read_wordvec(filename):

    import multiprocessing as mlp

    with open(wordvec[filename]) as f:
        wordmat=[line for line in f.readlines()]
        # wordmat=f.read().split('\n')
        if wordmat[-1]=='':
            wordmat = wordmat[:-1]
        if wordmat[0] == '':
            wordmat = wordmat[1:]

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

def bin_to_text(filename,name):
    from gensim.models.keyedvectors import KeyedVectors
    model = KeyedVectors.load_word2vec_format(PATH+filename, binary=True)
    model.save_word2vec_format(PATH+name,binary=False)


if __name__ == "__main__":
    # a=read_dataset('labels.csv')
    # a.info()
    # deal_index('labels.csv')
    bin_to_text('word2vec.bin','word2vec.txt')




