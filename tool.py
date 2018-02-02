import numpy as np
import random


def splitdata(index_train,dataset):
    train_x={}
    for key in dataset.keys():
        train_x[key] = dataset[key][index_train]
    return train_x


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






# class GenerateDataLoader()



















