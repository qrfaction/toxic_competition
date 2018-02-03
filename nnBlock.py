from keras.layers import Conv1D
from keras.layers import add
from keras.layers.pooling import MaxPool1D
import prepocess
import input
from keras.layers.advanced_activations import PReLU
from keras.layers.merge import concatenate
import tool
import numpy as np
import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from tqdm import tqdm

BATCHSIZE = 256

def CnnBlock(name,input_layer,filters):
    def Res_Inception(input_layer, filters, activate=True):
        filters = int(filters / 4)
        Ince_5 = Conv1D(filters=filters, kernel_size=1, strides=1, padding='same', activation='relu')(input_layer)
        Ince_5 = Conv1D(filters=filters, kernel_size=3, strides=1, padding='same', activation='relu')(Ince_5)
        Ince_5 = Conv1D(filters=filters, kernel_size=3, strides=1, padding='same')(Ince_5)
        Ince_5 = PReLU()(Ince_5)

        Ince_3 = Conv1D(filters=filters, kernel_size=1, strides=1, padding='same', activation='relu')(input_layer)
        Ince_3 = Conv1D(filters=filters, kernel_size=3, strides=1, padding='same')(Ince_3)
        Ince_3 = PReLU()(Ince_3)

        Ince_pool = MaxPool1D(pool_size=3, strides=1, padding='same')(input_layer)
        Ince_pool = Conv1D(filters=filters, kernel_size=1, strides=1, padding='same')(Ince_pool)
        Ince_pool = PReLU()(Ince_pool)

        Ince_1 = Conv1D(filters=filters, kernel_size=1, strides=1, padding='same')(input_layer)
        Ince_1 = PReLU()(Ince_1)

        Ince = concatenate([Ince_3, Ince_5, Ince_pool, Ince_1], axis=-1)
        if activate == True:
            res_util = add([input_layer, Ince])
            res_util = PReLU()(res_util)
        else:
            res_util = Ince
        return res_util

    def DenseNet(input_layer, filters ):
        filters = int(filters / 4)
        DBlock1 = Conv1D(filters=filters, kernel_size=1, strides=1, padding='same', activation='relu')(input_layer)
        DBlock1 = concatenate([DBlock1, input_layer])
        DBlock1 = Conv1D(filters=filters, kernel_size=1, strides=1, padding='same', activation='relu')(DBlock1)

        DBlock2 = Res_Inception(DBlock1, filters, activate=False)
        DBlock2 = concatenate([DBlock2, DBlock1, input_layer])
        DBlock2 = Conv1D(filters=filters, kernel_size=1, strides=1, padding='same', activation='relu')(DBlock2)

        DBlock3 = Res_Inception(DBlock2, filters, activate=False)
        DBlock3 = concatenate([DBlock3, DBlock2, DBlock1, input_layer])
        DBlock3 = Conv1D(filters=filters, kernel_size=1, strides=1, padding='same', activation='relu')(DBlock3)

        DBlock4 = Res_Inception(DBlock3, filters, activate=False)
        DBlock4 = concatenate([DBlock4, DBlock3, DBlock2, DBlock1])
        DBlock4 = add([DBlock4,input_layer])

        return DBlock4

    if name=='res_inception':
        return Res_Inception(input_layer,filters)
    elif name=='DenseNet':
        return DenseNet(input_layer,filters)

class auc_loss(nn.Module):
    def __init__(self,batchsize=BATCHSIZE//2):
        super(auc_loss,self).__init__()
        self.batchsize = batchsize

    def forward(self,y_pred,y_true):
        y_pred = torch.log(y_pred/(1-y_pred))
        loss = 0
        for i in range(6):
            yi = y_pred[:,i]
            yi_true = y_true[:,i]
            y_pos = torch.masked_select(yi,yi_true.gt(0.5))  # gt greater than
            y_pos = 1/y_pos
            y_neg = torch.masked_select(yi,yi_true.lt(0.5))  # lt less than

            if len(y_pos.size())==0 or len(y_neg.size())==0:
                continue
            mpos = y_pos.size()[0]
            mneg = y_neg.size()[0]
            y_pos = y_pos.unsqueeze(0)
            y_neg = y_neg.unsqueeze(1)
            loss += torch.sum(torch.mm(y_neg,y_pos))/(mpos*mneg)

        return loss

class focallogloss(nn.Module):
    def __init__(self,alpha):
        super(focallogloss, self).__init__()
        self.alpha = alpha
    def forward(self,y_pred,y_true):
        weight1 = torch.pow(1-y_pred,self.alpha)
        weight2 = torch.pow(y_pred,self.alpha)
        loss = -(
                    y_true * torch.log(y_pred) * weight1 +
                    (1-y_true) * torch.log(1-y_pred) * weight2
                )
        loss = torch.sum(loss)/(y_true.size()[0]*6)
        return  loss

class baseNet(nn.Module):
    def __init__(self,dim, embedding_matrix,trainable):
        super(baseNet,self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=len(embedding_matrix),
            embedding_dim=dim,
            padding_idx=0,
        )
        self.embedding.weight = nn.Parameter(torch.FloatTensor(embedding_matrix),requires_grad=trainable)
        self.GRU1 = nn.GRU(
            input_size=dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.GRU2 = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(64,6)

    def forward(self,sentences):

        x = self.embedding(sentences)
        hidden = autograd.Variable(torch.zeros(2,x.size()[0],128)).cuda()
        x,_ = self.GRU1(x,hidden)
        x = x[:,:,:128] + x[:,:,128:]
        hidden = autograd.Variable(torch.zeros(2, x.size()[0],64)).cuda()
        x,hn = self.GRU2(x,hidden)
        y = hn[0,:,:] + hn[1,:,:]
        y = y.squeeze()
        y = self.fc(y)
        y = nn.functional.sigmoid(y)
        return y



class DnnModle:

    def __init__(self,dim, embedding_matrix,alpha=2,trainable=True):
        super(DnnModle,self).__init__()

        self.basenet = baseNet(dim, embedding_matrix,trainable).cuda()

        self.optimizer = torch.optim.RMSprop(
            [
                {'params': self.basenet.GRU1.parameters()},
                {'params': self.basenet.GRU2.parameters()},
                {'params': self.basenet.fc.parameters()},
                {'params':self.basenet.embedding.parameters() ,'lr':1e-5},
            ],
            lr=0.001,
        )
        self.loss_f = auc_loss()
        self.loss_f2 = focallogloss(alpha=alpha)

    def fit(self,X,Y,loss='celoss'):
        comment = torch.autograd.Variable(torch.LongTensor(X['comment'].tolist()).cuda())

        Y = torch.autograd.Variable(torch.FloatTensor(Y.tolist()).cuda())
        y_pred = self.basenet(comment)
        if loss == 'celoss':
            loss = self.loss_f2(y_pred,Y)
        elif loss == 'auc loss':
            loss = self.loss_f(y_pred,Y)
        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def predict(self,X,batchsize=256):
        comment = torch.autograd.Variable(torch.LongTensor(X['comment'].tolist()).cuda(), volatile=True)
        y_pred = torch.zeros((len(comment),6)).cuda()
        for i in tqdm(range(0,len(comment),batchsize)):
            y_pred[i:i+batchsize] = self.basenet(comment[i:i+batchsize]).data
        return y_pred.cpu().numpy()




























