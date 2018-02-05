import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from tqdm import tqdm

BATCHSIZE = 256


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

        self.fc = nn.Linear(72,6)

        self.fc2 = nn.Linear(4,8)

    def forward(self,sentences,features):

        x = self.embedding(sentences)
        hidden = autograd.Variable(torch.zeros(2,x.size()[0],128)).cuda()
        x,_ = self.GRU1(x,hidden)
        x = x[:,:,:128] + x[:,:,128:]
        hidden = autograd.Variable(torch.zeros(2, x.size()[0],64)).cuda()
        x,hn = self.GRU2(x,hidden)
        y = hn[0,:,:] + hn[1,:,:]
        y = y.squeeze()
        features =self.fc2(features)
        features = nn.functional.sigmoid(features)
        y = torch.cat((y,features),1)
        y = self.fc(y)
        y = nn.functional.sigmoid(y)
        return y



class DnnModle:

    def __init__(self,dim, embedding_matrix,alpha=2,trainable=True,loss = 'focalLoss'):
        super(DnnModle,self).__init__()

        self.basenet = nn.DataParallel(baseNet(dim, embedding_matrix,trainable)).cuda()

        self.optimizer = torch.optim.RMSprop(
            [
                {'params': self.basenet.module.GRU1.parameters()},
                {'params': self.basenet.module.GRU2.parameters()},
                {'params': self.basenet.module.fc.parameters()},
                {'params': self.basenet.module.fc2.parameters()},
                {'params':self.basenet.module.embedding.parameters() ,'lr':1e-5},
            ],
            lr=0.001,
        )

        if loss == 'focalLoss':
            self.loss_f = focallogloss(alpha=alpha)
        elif loss =='aucLoss':
            self.loss_f = auc_loss()

    def fit(self,X,features,Y):
        comment = torch.autograd.Variable(X.cuda())
        Y = torch.autograd.Variable(Y.cuda())
        features = torch.autograd.Variable(features.cuda())
        y_pred = self.basenet(comment,features)
        loss = self.loss_f(y_pred,Y)
        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def predict(self,X,batchsize=2048):
        comment = torch.autograd.Variable(torch.LongTensor(X['comment'].tolist()).cuda(), volatile=True)
        features = torch.autograd.Variable(torch.FloatTensor(X['countFeature'].tolist()).cuda(),volatile=True)
        y_pred = torch.zeros((len(comment),6)).cuda()
        for i in tqdm(range(0,len(comment),batchsize)):
            y_pred[i:i+batchsize] = self.basenet(comment[i:i+batchsize],features[i:i+batchsize]).data
        return y_pred.cpu().numpy()





























