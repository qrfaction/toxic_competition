import re
import input
import string
from nltk.corpus import stopwords
import numpy as np
import multiprocessing as mlp
from tqdm import tqdm
from gensim.matutils import corpus2csc
from Ref_Data import replace_word
import pandas as pd
PATH = 'data/'

def countFeature(dataset):
    eng_stopwords = set(stopwords.words("english"))
    def CountFeatures(df):
        # 句子长度
        df['total_length'] = df['comment_text'].apply(len)

        # 大写字母个数
        df['capitals'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
        df['caps_vs_length'] = df.apply(lambda row: float(row['capitals']) / float(row['total_length']),
                                        axis=1)

        df['num_question_marks'] = df['comment_text'].apply(lambda comment: comment.count('?'))
        df['num_punctuation'] = df['comment_text'].apply(
            lambda comment: sum(comment.count(w) for w in '.,;:'))
        df['num_symbols'] = df['comment_text'].apply(
            lambda comment: sum(comment.count(w) for w in '*&$%'))
        df['num_words'] = df['comment_text'].apply(lambda comment: len(comment.split()))
        df['num_smilies'] = df['comment_text'].apply(
            lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))

        df['count_word'] = df["comment_text"].apply(lambda x: len(str(x).split()))
        df['count_unique_word'] = df["comment_text"].apply(lambda x: len(set(str(x).split())))
        df["count_punctuations"] = df["comment_text"].apply(
            lambda x: len([c for c in str(x) if c in string.punctuation]))
        df["count_stopwords"] = df["comment_text"].apply(
            lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
        df["mean_word_len"] = df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

        # derived features
        # 2个：非重复词占比、标点占比
        df['word_unique_percent'] = df['count_unique_word'] * 100 / df['count_word']
        df['punct_percent'] = df['count_punctuations'] * 100 / df['count_word']

        return df

    def LeakyFeatures(df):
        patternLink = '(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
        patternIP = '\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}'

        ## Leaky features——共8个特征
        df['ip'] = df["comment_text"].apply(lambda x: re.findall(patternIP, str(x)))
        df['count_ip'] = df["ip"].apply(lambda x: len(x))
        df['link'] = df["comment_text"].apply(lambda x: re.findall(patternLink, str(x)))
        df['count_links'] = df["link"].apply(lambda x: len(x))
        df['article_id'] = df["comment_text"].apply(lambda x: re.findall("\d:\d\d\s{0,5}$", str(x)))
        df['article_id_flag'] = df.article_id.apply(lambda x: len(x))
        df['username'] = df["comment_text"].apply(lambda x: re.findall("\[\[User(.*)\|", str(x)))
        df['count_usernames'] = df["username"].apply(lambda x: len(x))

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

    dataset["comment_text"] = dataset["comment_text"].apply(deal_space)
    dataset = CountFeatures(dataset)
    # dataset = LeakyFeatures(dataset)
    dataset = letter_distribution(dataset)
    return dataset


''' 封装TF-IDF '''


def tfidfFeature(clean_corpus, mode="other", params_tfidf=None, n_components=128):
    ''' TF-IDF Vectorizer '''
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    def getTfidfVector(clean_corpus,  # 之后的参数都是TfidfVectorizer()的参数
                       min_df=100, max_features=100000,
                       strip_accents='unicode', analyzer='word', ngram_range=(1, 1),
                       use_idf=1, smooth_idf=1, sublinear_tf=1,
                       stop_words='english'):

        tfv = TfidfVectorizer(min_df=min_df, max_features=max_features,
                              strip_accents=strip_accents, analyzer=analyzer, ngram_range=ngram_range,
                              use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf,
                              stop_words=stop_words)
        tfv.fit(clean_corpus)
        features_tfidf = np.array(tfv.get_feature_names())
        model_tfidf = tfv.transform(clean_corpus)
        return model_tfidf, features_tfidf

    ''' PCA降维 '''

    def pca_compression(model_tfidf, n_components):
        np_model_tfidf = model_tfidf.toarray()
        pca = PCA(n_components=n_components)
        pca_model_tfidf = pca.fit_transform(np_model_tfidf)
        return pca_model_tfidf

    ##### 确认模式 #####
    if mode == "other":
        # 初始化一套参数，然后用自定义的参数去替换更改后的
        params = {
            "min_df": 100, "max_features": 100000,
            "strip_accents": 'unicode', "analyzer": 'word', "ngram_range": (1, 1),
            "use_idf": 1, "smooth_idf": 1, "sublinear_tf": 1,
            "stop_words": 'english'
        }
        for item, value in params_tfidf.items():
            params[item] = params_tfidf[item]
    else:  # mode = "unigrams"/"bigrams"/"charngrams"
        ''' 内置3套参数 '''
        if mode == "unigrams":  # 单个词
            params = {
                "min_df": 100, "max_features": 100000,
                "strip_accents": 'unicode', "analyzer": 'word', "ngram_range": (1, 1),
                "use_idf": 1, "smooth_idf": 1, "sublinear_tf": 1,
                "stop_words": 'english'
            }
        elif mode == "bigrams":  # 两个词
            params = {
                "min_df": 100, "max_features": 30000,
                "strip_accents": 'unicode', "analyzer": 'word', "ngram_range": (2, 2),
                "use_idf": 1, "smooth_idf": 1, "sublinear_tf": 1,
                "stop_words": 'english'
            }
        elif mode == "charngrams":  # 长度为4的字符
            params = {
                "min_df": 100, "max_features": 30000,
                "strip_accents": 'unicode', "analyzer": 'char', "ngram_range": (1, 4),
                "use_idf": 1, "smooth_idf": 1, "sublinear_tf": 1,
                "stop_words": 'english'
            }
        else:
            print("mode error...")
            return

    # 获取tfidf后的稀疏矩阵sparse
    model_tfidf, features_tfidf = getTfidfVector(clean_corpus,  # 之后的参数都是TfidfVectorizer()的参数
                                                 min_df=params["min_df"], max_features=params["max_features"],
                                                 strip_accents=params["strip_accents"], analyzer=params["analyzer"],
                                                 ngram_range=params["ngram_range"],
                                                 use_idf=params["use_idf"], smooth_idf=params["smooth_idf"],
                                                 sublinear_tf=params["sublinear_tf"],
                                                 stop_words=params["stop_words"])
    # 获取pca后的np
    pca_model_tfidf = pca_compression(model_tfidf, n_components=n_components)
    # 获取添加特征名后的pd
    n = params["ngram_range"][0]  # 生成特征列名时的n的值
    pd_pca_model_tfidf = pd.DataFrame(pca_model_tfidf,
                                      columns=["tfidf" + str(n) + "gram" + str(x) for x in range(1, n_components + 1)])
    return pd_pca_model_tfidf

def doc2bow(text,dictionary):
    return [dictionary.doc2bow(t) for t in tqdm(text)]

def lda_infer(dataset,model):
    topic_probability_mat = model[dataset]
    return corpus2csc(topic_probability_mat).transpose().toarray().tolist()

def LDAFeature(num_topics=6):
    from embedding import tokenize_word
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


    text = [ [ word  for word in sentence if freq[word] > 4] for sentence in tqdm(text) ]

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


if __name__ == '__main__':
    from Ref_Data import NUM_TOPIC
    LDAFeature(NUM_TOPIC)
