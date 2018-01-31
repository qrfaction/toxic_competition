import numpy as np
import random


def splitdata(index_train,dataset):
    train_x={}
    for key in dataset.keys():
        train_x[key] = dataset[key][index_train]
    return train_x


class Generate:

    def __init__(self,train,labels):
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

    def genrerate_samples(self,col,batchsize=256):

        pos_list = []
        neg_list = []
        num = 0
        while num < batchsize :

            pos_index = random.choice(self.positive_samples[col])
            neg_index = random.choice(self.negative_samples[col])

            pair = (pos_index , neg_index) \
                if pos_index < neg_index else (neg_index,pos_index)

            if pair in self.history or pos_index == neg_index:
                continue

            self.history.add(pair)
            num += 2

            pos_list.append(pos_index)
            neg_list.append(neg_index)

        samples_index = pos_list.extend(neg_list)

        train_x = splitdata(samples_index,self.trainset)
        train_y = self.labels[samples_index]
        return train_x,train_y





















