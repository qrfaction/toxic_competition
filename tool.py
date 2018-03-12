import numpy as np
import random
from tqdm import tqdm
from textblob import TextBlob
from textblob.translate import NotTranslated
import json
import re
import input
from Ref_Data import BATCHSIZE
from random import randint

def translate(comments):
    translation = {}
    for id,comment in tqdm(comments):
        text = TextBlob(comment[0])
        try:
            text = text.translate(to="en")
        except NotTranslated:
            text = comment[0]
        translation[id] = [str(text),comment[0]]
    return translation

def deal_other_language():
    import multiprocessing as mlp
    with open('language_record.json') as f:
        comments = json.loads(f.read())

    results = []
    pool = mlp.Pool(mlp.cpu_count())
    aver_t = int(len(comments) / mlp.cpu_count()) + 1
    for i in range(mlp.cpu_count()):
        result = pool.apply_async(translate , args=(comments[i * aver_t:(i + 1) * aver_t],))
        results.append(result)
    pool.close()
    pool.join()

    translation = {}
    for result in results:
        translation.update(result.get())
    with open('translation.json', 'w') as f:
        f.write(json.dumps(translation, indent=4, separators=(',', ': '),ensure_ascii=False))

def get_other_lang_train(comments,lang='nl'):
    translation = []
    for comment in tqdm(comments):
        comment = re.sub("-", ' ', comment)
        text = TextBlob(comment)
        try:
            text = text.translate(to=lang)
        except NotTranslated:
            pass
        translation.append(str(text))
    return translation

def get_other_language_train(lang='nl'):
    import multiprocessing as mlp
    from Ref_Data import replace_word,PATH

    dataset = input.read_dataset('train.csv')
    dataset.fillna(replace_word['unknow'], inplace=True)
    comments = dataset['comment_text'].tolist()

    results = []
    pool = mlp.Pool(mlp.cpu_count())
    aver_t = int(len(comments) / mlp.cpu_count()) + 1
    for i in range(mlp.cpu_count()):
        result = pool.apply_async(get_other_lang_train,
                                  args=(comments[i * aver_t:(i + 1) * aver_t],lang))
        results.append(result)
    pool.close()
    pool.join()

    translation = []
    for result in results:
        translation.extend(result.get())
    dataset['comment_text'] = translation
    dataset.to_csv(PATH+lang+'_train.csv',index=False)

def splitdata(index_train,dataset):
    train_x={}
    for key in dataset.keys():
        train_x[key] = dataset[key][index_train]
    return train_x

class Generate:

    def __init__(self,train,labels,batchsize=BATCHSIZE,shuffle=True,window_size = 0):
        """
        :param labels: 标签  array  (samples,6)

        """
        self.labels = labels
        self.trainset = train
        self.seq_len = train['comment'].shape[1]

        self.positive_samples = {}
        self.negative_samples = {}
        for i in range(6):
            # where 返回元组
            self.positive_samples[str(i)] = np.where( labels[:,i]==1 )[0]
            self.negative_samples[str(i)] = np.where( labels[:,i]==0 )[0]

        self.batchsize = batchsize
        # sample
        self.begin = 0
        self.end   = self.batchsize
        self.index = list(range(0, len(labels)))
        if shuffle == True:
            np.random.shuffle(self.index)

        self.window_size = window_size

    def genrerate_rank_samples(self,col):

        samples_list = []
        num = 0
        while num < self.batchsize :

            pos_index = random.choice(self.positive_samples[col])
            neg_index = random.choice(self.negative_samples[col])

            num += 2

            samples_list.append(pos_index)
            samples_list.append(neg_index)


        train_x = splitdata(samples_list,self.trainset)

        train_y = self.labels[samples_list]
        return train_x,train_y

    def genrerate_balance_samples(self):
        col = random.choice([str(i) for i in range(6)])
        return self.genrerate_rank_samples(col)


    def get_randomPos(self):
        start1 = random.randint(0, self.seq_len - 2*self.window_size)
        end1 = start1 + self.window_size
        start2 = random.randint(end1 , self.seq_len - self.window_size)
        end2 = start2 + self.window_size
        return start1, end1, start2, end2


    def seq_noise(self,comments):
        for i in range(len(comments)):
            start1, end1, start2, end2 = self.get_randomPos()
            temp = comments[i,start1:end1]
            comments[i,start1:end1] = comments[i,start2:end2]
            comments[i,start2:end2] = temp
        return comments

    def genrerate_samples(self):

        sample_index = self.index[self.begin:self.end]

        train_x = splitdata(sample_index,self.trainset)
        train_y = self.labels[sample_index]
        self.begin = self.end
        self.end += self.batchsize
        if self.end > len(self.labels):
            np.random.shuffle(self.index)
            self.begin = 0
            self.end = self.batchsize

        if self.window_size>0:
            train_x['comment'] = self.seq_noise(train_x['comment'])
        return train_x,train_y

def cal_mean(results,scores=None):

    if scores is None:
        weights = np.ones((len(results),6))
    else :
        scores = np.array(scores)
        scores -= 0.980
        scores *= 10000
        weights = np.int64(scores)
        print(weights)

    test_predicts = np.zeros(results[0].shape)
    for i in range(6):
        for fold_predict,weight in zip(results,weights[:,i]):
            test_predicts[:,i] += (fold_predict[:,i] * weight)
        test_predicts[:,i] /= np.sum(weights[:,i])

    return test_predicts

def get_language():
    "检测语言是否为中文"
    from Ref_Data import replace_word
    import json
    from langdetect import detect_langs
    from langdetect.lang_detect_exception import LangDetectException
    train = input.read_dataset('train.csv').fillna(replace_word['unknow'])
    test  = input.read_dataset('test.csv').fillna(replace_word['unknow'])

    records = {}

    for index, row in tqdm(train.iterrows()):
        try:
            lang_prob = detect_langs(row['comment_text'])
            language = lang_prob[0].lang
            if language != 'en':
                records['tr' + str(index)] = (row['comment_text'], language, lang_prob[0].prob)
        except LangDetectException:
            records['tr' + str(index)] = (row['comment_text'], 'none',0)

    for index, row in tqdm(test.iterrows()):
        try:
            lang_prob = detect_langs(row['comment_text'])
            language = lang_prob[0].lang
            if language != 'en':
                records['te' + str(index)] = (row['comment_text'], language, lang_prob[0].prob)
        except LangDetectException:
            records['te' + str(index)] = (row['comment_text'], 'none',0)
    records = sorted(records.items(), key=lambda item: item[1][2], reverse=True)
    with open('language_record.json', 'w') as f:
        f.write(json.dumps(records, indent=4, separators=(',', ': '),ensure_ascii=False))

def add_comment(index,file):
    import input
    if file == 'te':
        dataset = input.read_dataset('test.csv')
    else:
        dataset = input.read_dataset('train.csv')
    with open('language_record.json') as f:
        comments = json.loads(f.read())
    for i in index:
        comment = [
            file+str(i),
            [
                dataset.loc[i,'comment_text'],
                "add",
                1
            ]
        ]
        comments.append(comment)
    with open('language_record.json', 'w') as f:
        f.write(json.dumps(comments, indent=4, separators=(',', ': '),ensure_ascii=False))


if __name__=="__main__":
    get_other_language_train('nl')   #荷兰语
    get_other_language_train('fr')   #法语















