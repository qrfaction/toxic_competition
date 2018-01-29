import numpy as np
import time
import warnings
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from Ref_Data import APPO
import input
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

warnings.filterwarnings("ignore")
PATH='data/'
UNKONW=' _UNK_ '

eng_stopwords = set(stopwords.words("english"))
lem = WordNetLemmatizer()


def CreateFeature(dataset):

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

        comment = re.sub("\\.+", ' . ', comment)

        comment = re.sub("\s+", " ", comment)

        return comment

    dataset["comment_text"] = dataset["comment_text"].apply(deal_space)
    dataset = CountFeatures(dataset)
    dataset = LeakyFeatures(dataset)

    return dataset

def cleanComment(comments):
    """
    This function receives comments and returns clean word-list
    """
    patternLink = '(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
    patternIP = '\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}'
    from enchant.tokenize import get_tokenizer
    import enchant

    tknzr = get_tokenizer("en_US")
    w_dict = enchant.Dict("en_US")

    clean_comments = []

    for comment in tqdm(comments):

        comment = comment.lower()
        # 去除IP
        comment = re.sub(patternIP, " ", comment)
        # 去除usernames
        comment = re.sub("\[\[.*\]", " ", comment)
        # 去除网址
        comment = re.sub(patternLink, " ", comment)
        # 去除非ascii字符
        comment = re.sub("[^\x00-\x7F]+", " ", comment)
        comment = re.sub("\\.+", ' . ', comment)
        # 分词
        words = word_tokenize(comment)

        # 省略词替换（参考APPO、nltk）：you're -> you are
        words = [APPO[word] if word in APPO else word for word in words]
        words = [lem.lemmatize(word, "v") for word in words]
        words = [w for w in words if w not in eng_stopwords]
        comment = " ".join(words)
        comment = comment.lower()

        comment = re.sub('[\|=*/\'\`\~]+',' ',comment)
        comment = re.sub('\p{P}+','.',comment)
        comment = re.sub('\\.+', ' . ', comment)
        comment = re.sub('\s+',' ',comment)

        # 纠正拼写错误
        # for word,pos in tknzr(comment):
        #     if w_dict.check(word) == False:
        #         try:
        #             comment = comment[:pos] + \
        #                       w_dict.suggest(word)[0] + \
        #                       comment[pos+len(word):]
        #             print(word,w_dict.suggest(word)[0])
        #         except IndexError:
        #             continue
        clean_comments.append(comment)
    return clean_comments

def clean_dataset(dataset,filename):
    import multiprocessing as mlp

    results = []
    pool = mlp.Pool(mlp.cpu_count())

    comments = list(dataset['comment_text'])
    aver_t = int(len(dataset) / mlp.cpu_count()) + 1
    for i in range(mlp.cpu_count()):
        result = pool.apply_async(cleanComment, args=(comments[i * aver_t:(i + 1) * aver_t],))
        results.append(result)
    pool.close()
    pool.join()

    clean_comments = []
    for result in results:
        clean_comments.extend(result.get())

    dataset['comment_text'] = clean_comments

    dataset.to_csv(PATH+filename,index=False)

def getTfidfVector(clean_corpus):
    '''
    TF-IDF Vectorizer
    '''
    ### 单个词 ###
    tfv = TfidfVectorizer(min_df=100,  max_features=100000, 
                strip_accents='unicode', analyzer='word',ngram_range=(1,1),
                use_idf=1,smooth_idf=1,sublinear_tf=1,
                stop_words = 'english')
    tfv.fit(clean_corpus)
#    features = np.array(tfv.get_feature_names())
    
    train_unigrams =  tfv.transform(clean_corpus.iloc[:train.shape[0]])
#    test_unigrams = tfv.transform(clean_corpus.iloc[train.shape[0]:])
    
    print("total time till unigrams",time.time()-start_time)
    
    ### 两个词 ###
    tfv = TfidfVectorizer(min_df=100,  max_features=30000, 
                strip_accents='unicode', analyzer='word',ngram_range=(2,2),
                use_idf=1,smooth_idf=1,sublinear_tf=1,
                stop_words = 'english')
    
    tfv.fit(clean_corpus)
#    features = np.array(tfv.get_feature_names())
    train_bigrams =  tfv.transform(clean_corpus.iloc[:train.shape[0]])
#    test_bigrams = tfv.transform(clean_corpus.iloc[train.shape[0]:])
    
    print("total time till bigrams",time.time()-start_time)
    
    ### 长度为4的字符 ###
    tfv = TfidfVectorizer(min_df=100,  max_features=30000, 
                strip_accents='unicode', analyzer='char',ngram_range=(1,4),
                use_idf=1,smooth_idf=1,sublinear_tf=1,
                stop_words = 'english')
    
    tfv.fit(clean_corpus)
#    features = np.array(tfv.get_feature_names())
    train_charngrams =  tfv.transform(clean_corpus.iloc[:train.shape[0]])
#    test_charngrams = tfv.transform(clean_corpus.iloc[train.shape[0]:])

    
    return train_bigrams,train_charngrams,train_unigrams

def splitTarget(filename):
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    labels=input.read_dataset(filename,list_classes)
    labels.to_csv(PATH+'labels.csv',index=False)

def pipeline():
    file = ['train.csv','test.csv','train_fr.csv','train_es.csv','train_de.csv']
    for filename in tqdm(file):
        dataset = input.read_dataset(filename)
        dataset.fillna(UNKONW,inplace=True)
        dataset = CreateFeature(dataset)
        clean_dataset(dataset,'clean_'+filename)



if __name__ == "__main__":
    pipeline()
