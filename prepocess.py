# -*- coding: utf-8 -*-
# 复制所有代码，精简到最简版
##### 导入包 #####
#basics
import pandas as pd 
import numpy as np
import time
import warnings
import seaborn as sns
import string
import re    #for regex
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from Ref_Data import APPO
import input
#settings

sns.set_style("dark")
eng_stopwords = set(stopwords.words("english"))
warnings.filterwarnings("ignore")

lem = WordNetLemmatizer()
tokenizer = TweetTokenizer()
PATH='data/'
##### 特征工程 #####
def makeIndirectFeatures(train, df):
    ## Indirect features——共11个特征
    #9个：单词句子数、数、非重复单词数、字母数、标点数、大写字母的单词/字母数、标题数、停顿词数、单词平均长度
    df['count_sent']=df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)
    df['count_word']=df["comment_text"].apply(lambda x: len(str(x).split()))
    df['count_unique_word']=df["comment_text"].apply(lambda x: len(set(str(x).split())))
    df['count_letters']=df["comment_text"].apply(lambda x: len(str(x)))
    df["count_punctuations"] =df["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    df["count_words_upper"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    df["count_words_title"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    df["count_stopwords"] = df["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
    df["mean_word_len"] = df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    
    #derived features
    #2个：非重复词占比、标点占比
    df['word_unique_percent']=df['count_unique_word']*100/df['count_word']
    df['punct_percent']=df['count_punctuations']*100/df['count_word']
    
    #serperate train and test features
    train_feats=df.iloc[0:len(train),]
#    test_feats=df.iloc[len(train):,]
    #join the tags
    train_tags=train.iloc[:,2:]
    train_feats=pd.concat([train_feats,train_tags],axis=1)
    
    # 限定值的范围
    train_feats['count_sent'].loc[train_feats['count_sent']>10] = 10 
    train_feats['count_word'].loc[train_feats['count_word']>200] = 200
    train_feats['count_unique_word'].loc[train_feats['count_unique_word']>200] = 200
    return train_feats, train_tags
    
def makeLeakyFeatures(train, df):
    ## Leaky features——共8个特征
    df['ip']=df["comment_text"].apply(lambda x: re.findall("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}",str(x)))
    df['count_ip']=df["ip"].apply(lambda x: len(x))
    df['link']=df["comment_text"].apply(lambda x: re.findall("http://.*com",str(x)))
    df['count_links']=df["link"].apply(lambda x: len(x))
    df['article_id']=df["comment_text"].apply(lambda x: re.findall("\d:\d\d\s{0,5}$",str(x)))
    df['article_id_flag']=df.article_id.apply(lambda x: len(x))
    df['username']=df["comment_text"].apply(lambda x: re.findall("\[\[User(.*)\|",str(x)))
    df['count_usernames']=df["username"].apply(lambda x: len(x))
    #check if features are created
    #df.username[df.count_usernames>0]
    
    leaky_feats = df[["ip","link","article_id","username","count_ip","count_links","count_usernames","article_id_flag"]]
    leaky_feats_train = leaky_feats.iloc[:train.shape[0]]
#    leaky_feats_test=leaky_feats.iloc[train.shape[0]:]
    return leaky_feats_train

def clean(comment):
    """
    This function receives comments and returns clean word-list
    """
    #小写化：Hi与hi等同
    comment=comment.lower()
    #去除\n
    comment=re.sub("\\n+",".",comment)
    #去除IP
    comment=re.sub("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}","",comment)
    #去除usernames
    comment=re.sub("\[\[.*\]","",comment)
    #去除网址
    comment=re.sub("(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]"," ",comment)
    #去除非ascii字符
    comment = re.sub("[^\x00-\x7F]+", "", comment)

    #分离句子为单词
    words=tokenizer.tokenize(comment)
    
    # 省略词替换（参考APPO、nltk）：you're -> you are  
    words=[APPO[word] if word in APPO else word for word in words]
    words=[lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if not w in eng_stopwords]
    
    clean_sent=" ".join(words)
    # remove any non alphanum,digit character
    # clean_sent=re.sub("\W+"," ",clean_sent)
    clean_sent=re.sub("\s+"," ",clean_sent)
    return(clean_sent)

def clean_dataset(filename):
    dataset=input.read_dataset(filename,["id","comment_text"])
    dataset.fillna('UNKNOW',inplace=True)
    dataset["comment_text"]=dataset["comment_text"].apply(clean)
    dataset.to_csv(PATH+'clean_'+filename,index=False)
    print(dataset)

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

def comment_to_seq(train,test,maxlen=100,dimension=300,wordvecfile='glove42'):
    from keras.preprocessing.text import Tokenizer
    train.fillna(' ',inplace=True)
    test.fillna(' ',inplace=True)
    text=train['comment_text'].values.tolist()+test['comment_text'].values.tolist()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)

    sequences = tokenizer.texts_to_sequences(text)
    sequences = pad_sequences(sequences,maxlen=maxlen,truncating='post')
    trainseq=sequences[:len(train)]
    testseq=sequences[len(train):]
    assert len(trainseq) == len(train)
    assert len(testseq) == len(test)

    word_index=tokenizer.word_index

    num_words =len(word_index)
    # embedding_matrix = np.random.normal(loc=0.0, scale=0.33,size=(num_words+1,dimension))
    embedding_matrix = np.zeros((num_words+1,dimension))
    embedding_matrix[0]= 0
    embeddings_index = input.read_wordvec(wordvecfile)

    noword=0
    for word, i in tqdm(word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is  None:
            noword+=1
        else:
            embedding_matrix[i] = embedding_vector
    print(noword)
    return trainseq,testseq,embedding_matrix

def splitTarget(filename):
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    labels=input.read_dataset(filename,list_classes)
    labels.to_csv(PATH+'labels.csv',index=False)


if __name__ == "__main__":
    clean_dataset('train.csv')
    clean_dataset('test.csv')
