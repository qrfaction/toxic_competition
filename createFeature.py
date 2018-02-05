import re
import input
import string
from nltk.corpus import stopwords
import numpy as np
import multiprocessing as mlp
from tqdm import tqdm
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
        df['num_exclamation_marks'] = df['comment_text'].apply(lambda comment: comment.count('!'))
        df['num_question_marks'] = df['comment_text'].apply(lambda comment: comment.count('?'))
        df['num_punctuation'] = df['comment_text'].apply(
            lambda comment: sum(comment.count(w) for w in '.,;:'))
        df['num_symbols'] = df['comment_text'].apply(
            lambda comment: sum(comment.count(w) for w in '*&$%'))
        df['num_words'] = df['comment_text'].apply(lambda comment: len(comment.split()))
        df['num_unique_words'] = df['comment_text'].apply(
            lambda comment: len(set(w for w in comment.split())))
        df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
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

    def deal_space(comment):

        comment = re.sub("\\n+", ".", comment)

        comment = re.sub("\.{2,}", ' . ', comment)

        comment = re.sub("\s+", " ", comment)

        return comment

    dataset["comment_text"] = dataset["comment_text"].apply(deal_space)
    dataset = CountFeatures(dataset)
    dataset = LeakyFeatures(dataset)

    return dataset

def getTfidfVector(train,test):
    '''
    TF-IDF Vectorizer
    '''
    from sklearn.feature_extraction.text import TfidfVectorizer

    ### 单个词 ###
    tfv = TfidfVectorizer(min_df=100, max_features=50000,
                          strip_accents='unicode', analyzer='word', ngram_range=(1, 1),
                          use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')
    tfv.fit(clean_corpus)
    #    features = np.array(tfv.get_feature_names())

    train_unigrams = tfv.transform(clean_corpus.iloc[:train.shape[0]])
    #    test_unigrams = tfv.transform(clean_corpus.iloc[train.shape[0]:])


    ### 两个词 ###
    tfv = TfidfVectorizer(min_df=100, max_features=50000,
                          strip_accents='unicode', analyzer='word', ngram_range=(2, 2),
                          use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')

    tfv.fit(clean_corpus)
    #    features = np.array(tfv.get_feature_names())
    train_bigrams = tfv.transform(clean_corpus.iloc[:train.shape[0]])
    #    test_bigrams = tfv.transform(clean_corpus.iloc[train.shape[0]:])


    ### 长度为4的字符 ###
    tfv = TfidfVectorizer(min_df=100, max_features=50000,
                          strip_accents='unicode', analyzer='char', ngram_range=(1, 4),
                          use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')

    tfv.fit(clean_corpus)
    #    features = np.array(tfv.get_feature_names())
    train_charngrams = tfv.transform(clean_corpus.iloc[:train.shape[0]])
    #    test_charngrams = tfv.transform(clean_corpus.iloc[train.shape[0]:])


    return train_bigrams, train_charngrams, train_unigrams

def doc2bow(text,dictionary):
    return [dictionary.doc2bow(t) for t in tqdm(text)]



def LDAFeature(num_topics=20):
    from embedding import tokenize_word
    from gensim.corpora import Dictionary
    from gensim.matutils import corpus2csc
    from gensim.models.ldamulticore import LdaMulticore

    def get_corpus(dictionary):
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

    train = input.read_dataset('clean_train.csv')
    test = input.read_dataset('clean_test.csv')
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

    corpus = get_corpus(dictionary)
    print('begin train lda')
    ldamodel = LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary)

    print('inference')
    topic_probability_mat = ldamodel[corpus]


    train_matrix = topic_probability_mat[:train.shape[0]]
    test_matrix = topic_probability_mat[train.shape[0]:]

    train_sparse = corpus2csc(train_matrix).transpose().toarray()
    test_sparse = corpus2csc(test_matrix).transpose().toarray()

    effective_section = {}
    for topics in tqdm(train_sparse):
        num = np.sum(topics==0)
        num =str(int(num))
        if num not in effective_section:
            effective_section[num] = 0
        effective_section[num]+=1
    print(effective_section)


    print('save')
    for i in range(num_topics):
        train['topic'+str(i)] = 0
        test['topic'+str(i)] = 0
    train[['topic'+str(i) for i in range(num_topics)]] = train_sparse
    test[['topic' + str(i) for i in range(num_topics)]] = test_sparse

    train.to_csv(PATH+'clean_train.csv',index=False)
    test.to_csv(PATH + 'clean_test.csv', index=False)


if __name__ == '__main__':
    LDAFeature()
