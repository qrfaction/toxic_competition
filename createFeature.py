import re
import input
import numpy as np
import multiprocessing as mlp
from tqdm import tqdm
from gensim.matutils import corpus2csc
from Ref_Data import replace_word,FILTER_FREQ,NUM_TOPIC,POSTAG_DIM,PATH,CHAR_N
import pandas as pd
from sklearn.decomposition import PCA,KernelPCA,SparsePCA
from sklearn.feature_extraction.text import TfidfVectorizer
from embedding import tokenize_word,batch_char_analyzer




def countFeature(dataset):
    def CountFeatures(df):
        # 句子长度
        df['total_length'] = df['comment_text'].apply(lambda x:min(len(x),200*4))
        # 大写字母个数
        df['capitals'] = df['comment_text'].apply(lambda x: min(sum(1 for c in x if c.isupper()),20))
        df['caps_vs_length'] = df['capitals']/ df['total_length']
        df['num_words'] = df['comment_text'].apply(lambda x: min(len(x.split()),200))

        df['count_unique_word'] = df["comment_text"].apply(lambda x:
                                                           min(len(set(str(x).split())) ,200))
        df["mean_word_len"] = df["comment_text"].apply(lambda x: min(np.mean([len(w) for w in str(x).split()]),10))

        return df

    def letter_distribution(df):
        for i in range(97,97+26):
            df['distri_'+chr(i)] = df['comment_text'].apply(lambda comment: comment.count(chr(i)))
        df['distri_'+'!'] = df['comment_text'].apply(lambda comment: comment.count('!'))

        columns = ['distri_'+chr(i) for i in range(97,97+26)]
        columns.append('distri_!')
        def normalize(comment):
            comment[columns] =  comment[columns]/(comment[columns].sum()+0.01)
            return comment
        df = df.apply(normalize,axis=1)
        return df

    def deal_space(comment):

        comment = re.sub("\\n+", ".", comment)

        comment = re.sub("\.{2,}", ' . ', comment)

        comment = re.sub("\s+", " ", comment)

        return comment
    dataset["comment_text"]=dataset["comment_text"].fillna(replace_word['unknow'])
    dataset['count_sent'] = dataset["comment_text"].apply(lambda x: min(len(re.findall("\n", str(x))) + 1,10))
    dataset["comment_text"] = dataset["comment_text"].apply(deal_space)
    dataset = CountFeatures(dataset)
    dataset = letter_distribution(dataset)
    return dataset

''' 封装TF-IDF '''

def tfidfFeature(n_components=CHAR_N):
    ''' TF-IDF Vectorizer '''
    train = input.read_dataset('clean_train.csv')
    test = input.read_dataset('clean_test.csv')
    train['comment_text'] = train['comment_text'].fillna(replace_word['unknow'])
    test['comment_text'] = test['comment_text'].fillna(replace_word['unknow'])
    text = train['comment_text'].values.tolist() + test['comment_text'].values.tolist()

    def pca_compression(model_tfidf, n_components):
        np_model_tfidf = model_tfidf.toarray()
        pca = PCA(n_components=n_components)
        pca_model_tfidf = pca.fit_transform(np_model_tfidf)
        return pca_model_tfidf


    tfv = TfidfVectorizer(min_df=100, max_features=30000,
                          strip_accents='unicode', analyzer='char', ngram_range= (2, 4),
                          use_idf=1, smooth_idf=True, sublinear_tf=True)
    model_tfidf = tfv.fit_transform(text)

    # 获取pca后的np
    pca_model_tfidf = pca_compression(model_tfidf, n_components=n_components)
    # 获取添加特征名后的pd
    print(pca_model_tfidf.shape)
    cols = ["tfidf" + str(x) for x in range( n_components)]
    pca_model_tfidf = pd.DataFrame(pca_model_tfidf,columns=cols)

    for col in cols:
        pca_model_tfidf[col] = \
            (pca_model_tfidf[col]-pca_model_tfidf[col].mean())/pca_model_tfidf[col].std()
        list_col = pca_model_tfidf[col].tolist()
        train[col] = list_col[:len(train)]
        test[col] = list_col[len(train):]

    print('save')
    train.to_csv(PATH + 'clean_train.csv', index=False)
    test.to_csv(PATH + 'clean_test.csv', index=False)



def doc2bow(text,dictionary):
    return [dictionary.doc2bow(t) for t in tqdm(text)]

def lda_infer(dataset,model):
    topic_probability_mat = model[dataset]
    return corpus2csc(topic_probability_mat).transpose().toarray().tolist()

def LDAFeature(num_topics=NUM_TOPIC):
    from gensim.corpora import Dictionary
    from gensim.models.ldamulticore import LdaMulticore

    def get_corpus(dictionary,text):
        results = []
        pool = mlp.Pool(mlp.cpu_count())

        comments = list(text)
        aver_t = int(len(text) / mlp.cpu_count()) + 1
        for i in range(mlp.cpu_count()):
            result = pool.apply_async(doc2bow, args=(comments[i*aver_t : (i + 1)*aver_t],dictionary))
            results.append(result)
        pool.close()
        pool.join()

        corpus = []
        for result in results:
            corpus.extend(result.get())
        return corpus

    def inference(model,dataset):
        results = []
        pool = mlp.Pool(mlp.cpu_count())

        aver_t = int(len(dataset) / mlp.cpu_count()) + 1
        for i in range(mlp.cpu_count()):
            result = pool.apply_async(lda_infer, args=(dataset[i * aver_t: (i + 1) * aver_t],model))
            results.append(result)
        pool.close()
        pool.join()

        topics = []
        for result in results:
            topics.extend(result.get())
        return np.array(topics)

    train = input.read_dataset('clean_train.csv')
    test = input.read_dataset('clean_test.csv')
    train['comment_text'] = train['comment_text'].fillna(replace_word['unknow'])
    test['comment_text'] = test['comment_text'].fillna(replace_word['unknow'])
    text = train['comment_text'].values.tolist() + test['comment_text'].values.tolist()

    text = tokenize_word(text)

    freq = {}
    for sentence in text:
        for word in sentence:
            if word not in freq:
                freq[word] = 0
            freq[word] +=1


    text = [ [ word  for word in sentence if freq[word] > FILTER_FREQ] for sentence in tqdm(text) ]

    dictionary = Dictionary(text)     # 生成 (id,word) 字典

    corpus = get_corpus(dictionary,text)
    print(len(corpus),len(corpus[0]))
    print('begin train lda')
    ldamodel = LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary)

    print('inference')
    topic_probability_mat = inference(ldamodel,corpus)
    print(len(topic_probability_mat),len(topic_probability_mat[0]))

    train_sparse = topic_probability_mat[:train.shape[0]]
    test_sparse = topic_probability_mat[train.shape[0]:]



    # 计算有效成分有多少
    zero_section = {}
    for topics in tqdm(train_sparse):
        num = np.sum(topics==0)
        num =str(int(num))
        if num not in zero_section:
            zero_section[num] = 0
        zero_section[num]+=1
    for topics in tqdm(test_sparse):
        num = np.sum(topics==0)
        num =str(int(num))
        if num not in zero_section:
            zero_section[num] = 0
        zero_section[num]+=1
    print(zero_section)


    print('save')
    for i in range(num_topics):
        train['topic'+str(i)] = 0
        test['topic'+str(i)] = 0
    train[['topic'+str(i) for i in range(num_topics)]] = train_sparse
    test[['topic' + str(i) for i in range(num_topics)]] = test_sparse

    train.to_csv(PATH+'clean_train.csv',index=False)
    test.to_csv(PATH + 'clean_test.csv', index=False)

def get_tag(text,pos_tag):
    result = []
    word2tag = {}
    for t in tqdm(text):
        text_tag = []
        for word,tag in pos_tag(t):
            text_tag.append(tag.lower())
            word2tag[word] = tag.lower()
        text_tag = ' '.join(text_tag)
        result.append(text_tag)
    return result,word2tag

def get_pos_tag_vec():
    from nltk import pos_tag
    train = input.read_dataset('clean_train.csv')
    test = input.read_dataset('clean_test.csv')
    train['comment_text'] = train['comment_text'].fillna(replace_word['unknow'])
    test['comment_text'] = test['comment_text'].fillna(replace_word['unknow'])
    text = train['comment_text'].values.tolist() + test['comment_text'].values.tolist()
    text = tokenize_word(text)

    def get_tag_text(text):
        results = []
        pool = mlp.Pool(mlp.cpu_count())

        comments = list(text)
        aver_t = int(len(text) / mlp.cpu_count()) + 1
        for i in range(mlp.cpu_count()):
            result = pool.apply_async(get_tag, args=(comments[i * aver_t: (i + 1) * aver_t],pos_tag))
            results.append(result)
        pool.close()
        pool.join()

        text_tag = []
        word2tag = {}
        for result in results:
            t_tag,word_2_vec = result.get()
            text_tag.extend(t_tag)
            word2tag.update(word_2_vec)
        return text_tag,word2tag

    def getTfidfVector(clean_corpus,
                       min_df=0,max_features=int(1e10),
                       ngram_range=(1, 1),use_idf=False,sublinear_tf=True):
        def tokenizer(t):
            return t.split()
        tfv = TfidfVectorizer(min_df=min_df, max_features=max_features,tokenizer=tokenizer,
                              strip_accents=None, analyzer="word", ngram_range=ngram_range,
                              use_idf=use_idf, sublinear_tf=sublinear_tf)
        tag_tfidf = tfv.fit_transform(clean_corpus)
        return tag_tfidf,list(tfv.get_feature_names())


    text_tag,word2tag = get_tag_text(text)
    import json

    with open(PATH+'word2tag.json', 'w') as f:
        f.write(json.dumps(word2tag , indent=4, separators=(',', ': ')))

    tag_tfidf ,columns= getTfidfVector(text_tag)
    n_components = POSTAG_DIM   # 输出pca.lambda_ 选择99%的成分即可
    pca = KernelPCA(n_components=n_components,kernel='rbf',n_jobs=-1)
    pca_tfidf = pca.fit_transform(tag_tfidf.transpose()).transpose()

    postag_vec = pd.DataFrame(pca_tfidf,columns=columns)
    postag_vec.to_csv(PATH+'postagVec.csv',index=False)

def createKmeansFeature(usecols,name,k=6):
    from sklearn.cluster import KMeans
    train = input.read_dataset('clean_train.csv')
    test = input.read_dataset('clean_test.csv')
    data = train.append(test)[usecols].values

    # def distMeas(vecA, vecB):
    #     return np.sqrt(np.sum(np.power(vecA - vecB, 2), axis=1))
    #
    # def KMeans(dataSet, k):
    #     """
    #     k-means 聚类算法
    #     该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。这个过程重复数次，直到数据点的簇分配结果不再改变为止。
    #     """
    #     def createRandCent(dataSet, k):
    #         """
    #         为给定数据集构建一个包含k个随机质心的集合。
    #         """
    #         n = dataSet.shape[1]  # 列的数量
    #         feature_min = dataSet.min(axis=0)  # 获取每个特征的下界
    #         feature_range = dataSet.max(axis=0) - feature_min
    #         centroids = feature_min + feature_range * np.random.random((k, n))
    #         return centroids
    #
    #     m = dataSet.shape[0]  # 行数
    #     clusterAssment = np.zeros(m)  # 创建一个与 dataSet 行数一样，但是有两列的矩阵，用来保存簇分配结果（一列簇索引值、一列误差）
    #     centroids = createRandCent(dataSet, k)  # 创建质心，随机k个质心
    #     distance = np.zeros((m, k))
    #     clusterChanged = True
    #     while clusterChanged:
    #         for j in range(k):
    #             distance[:, j] = distMeas(centroids[j, :], dataSet)
    #
    #         sample_cluster = distance.argmin(axis=1)  # 获取所属的簇
    #         num_change = np.sum(clusterAssment != sample_cluster)  # 有多少样本所属簇变了
    #         if num_change == 0:
    #             clusterChanged = False
    #         clusterAssment = sample_cluster
    #
    #         for center in range(k):  # 更新质心的位置
    #             ptsInClust = dataSet[clusterAssment == center]  # 获取该簇中的所有点
    #             centroids[center, :] = np.mean(ptsInClust, axis=0)
    #         # 处理nan
    #         centroids = np.nan_to_num(centroids)
    #     return centroids

    # samples = data[usecols].values
    # centroids = KMeans(samples ,k)       # kMeans聚类

    # for j in range(k):  # k为质心数
    #     data["kmeans" + str(j + 1)] = \
    #         distMeas(centroids[j, :], samples)  # 计算数据点到各个质心的距离

    model = KMeans(6,max_iter=3000,tol=1e-6,n_jobs=-1)
    features = model.fit_transform(data)
    for i in range(k):
        train[name+'_kmean_'+str(i)] = features[:len(train),i]
        test[name+'_kmean_'+str(i)] = features[len(train):, i]

    train.to_csv(PATH + 'clean_train.csv', index=False)
    test.to_csv(PATH + 'clean_test.csv', index=False)

def get_char_text():
    train = input.read_dataset('clean_train.csv')
    test = input.read_dataset('clean_test.csv')
    train['comment_text'] = train['comment_text'].fillna(replace_word['unknow'])
    test['comment_text'] = test['comment_text'].fillna(replace_word['unknow'])
    text = train['comment_text'].values.tolist() + test['comment_text'].values.tolist()
    text = tokenize_word(text)

    def get_ch_seqs(text):
        results = []
        pool = mlp.Pool(mlp.cpu_count())

        comments = list(text)
        aver_t = int(len(text) / mlp.cpu_count()) + 1
        for i in range(mlp.cpu_count()):
            result = pool.apply_async(batch_char_analyzer,
                            args=(comments[i * aver_t: (i + 1) * aver_t],True))
            results.append(result)
        pool.close()
        pool.join()

        ch_seqs = []
        for result in results:
            char_seq = result.get()
            ch_seqs.extend(char_seq)

        return ch_seqs

    seqs = get_ch_seqs(text)
    train['char_text'] = seqs[:len(train)]
    test['char_text'] = seqs[len(train):]
    train.to_csv(PATH + 'clean_train.csv', index=False)
    test.to_csv(PATH + 'clean_test.csv', index=False)

def char2idx(wordvecfile):
    train = input.read_dataset('clean_train.csv')
    test = input.read_dataset('clean_test.csv')
    train['comment_text'] = train['comment_text'].fillna(replace_word['unknow'])
    test['comment_text'] = test['comment_text'].fillna(replace_word['unknow'])
    text = train['comment_text'].values.tolist() + test['comment_text'].values.tolist()
    text = tokenize_word(text)
    input.read_wordvec(wordvecfile)
    def get_ch_seqs(text):
        results = []
        pool = mlp.Pool(mlp.cpu_count())

        comments = list(text)
        aver_t = int(len(text) / mlp.cpu_count()) + 1
        for i in range(mlp.cpu_count()):
            result = pool.apply_async(batch_char_analyzer,
                                      args=(comments[i * aver_t: (i + 1) * aver_t], True))
            results.append(result)
        pool.close()
        pool.join()

        ch_seqs = []
        for result in results:
            char_seq = result.get()
            ch_seqs.extend(char_seq)

        return ch_seqs

    import itertools,json
    corpus_chars = list(itertools.chain.from_iterable(corpus_chars)) #2维list展开成1维
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    with open(PATH+'char2index.json', 'w') as f:
        f.write(json.dumps(char_to_idx, indent=4, separators=(',', ': ')))

if __name__ == '__main__':
    tfidfFeature(CHAR_N)
