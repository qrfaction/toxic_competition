import tensorflow     #core dump 需要
import fastText       #core dump 需要
import numpy as np
import random
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from textblob import TextBlob
from textblob.translate import NotTranslated
import json

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

def splitdata(index_train,dataset):
    train_x={}
    for key in dataset.keys():
        train_x[key] = dataset[key][index_train]
    return train_x


class CommentData(Dataset):
    def __init__(self, trainset , labels):
        self.trainset = torch.LongTensor(trainset['comment'].tolist())
        self.labels = torch.FloatTensor(labels.tolist())

        self.features = torch.FloatTensor(trainset['countFeature'].tolist())

    def __getitem__(self, index):#返回的是tensor
        return self.trainset[index],self.features[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class Generate:

    def __init__(self,train,labels,batchsize=256,shuffle=True):
        """
        :param labels: 标签  array  (samples,6)

        """
        self.labels = labels
        self.trainset = train

        self.positive_samples = {}
        self.negative_samples = {}
        for i in range(6):
            # where 返回元组
            self.positive_samples[i] = np.where( labels[:,i]==1 )[0]
            self.negative_samples[i] = np.where( labels[:,i]==0 )[0]
        self.history = set([])

        self.batchsize = batchsize
        # sample
        self.begin = 0
        self.end   = self.batchsize
        self.index = list(range(0, len(labels)))
        if shuffle == True:
            np.random.shuffle(self.index)


    def genrerate_rank_samples(self,col):

        samples_list = []
        num = 0
        while num < self.batchsize :

            pos_index = random.choice(self.positive_samples[col])
            neg_index = random.choice(self.negative_samples[col])

            pair = (pos_index , neg_index) \
                if pos_index < neg_index else (neg_index,pos_index)

            if pair in self.history or pos_index == neg_index:
                continue

            self.history.add(pair)
            num += 2

            samples_list.append(pos_index)
            samples_list.append(neg_index)


        train_x = splitdata(samples_list,self.trainset)

        train_y = self.labels[samples_list]
        return train_x,train_y

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
        return train_x,train_y


def cal_mean(results,scores=None):

    if scores is None:
        weights = np.ones((len(results),6))
    else :
        scores = np.array(scores)
        scores -= 0.98
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
    import input
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
    # get_language()

    index = [31903,32494,109104]
    add_comment(index,'te')
    deal_other_language()















