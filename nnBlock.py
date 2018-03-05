import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from tqdm import tqdm
from Ref_Data import BATCHSIZE,model_setting
from Nadam import Nadam

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
        self.dropout = nn.Dropout(p=0.3)
        self.GRU1 = nn.GRU(
            input_size=dim,
            hidden_size=model_setting['hidden_size1'],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.5,
        )

        self.GRU2 = nn.GRU(
            input_size=model_setting['hidden_size1'],
            hidden_size=model_setting['hidden_size2'],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.5,
        )

        # self.atte_fc1 = nn.Conv1d(200,100,kernel_size=1)
        # self.atte_fc2 = nn.Conv1d(200,1,kernel_size=1)

        self.maxPool = nn.MaxPool1d(200)
        self.avePool = nn.AvgPool1d(200)

        from Ref_Data import NUM_TOPIC,USE_LETTERS,USE_TOPIC
        dim_features = 4
        if USE_TOPIC:
            dim_features += NUM_TOPIC
        if USE_LETTERS:
            dim_features +=27
        # self.fc2 = nn.Linear(dim_features  ,model_setting['fc_feature'])

        dim_fc = model_setting['hidden_size2']*2 + dim_features
        self.fc = nn.Linear(dim_fc ,6)


    def forward(self,sentences,features,volatile):

        x = self.embedding(sentences)
        # hidden = autograd.Variable(torch.zeros(2,x.size()[0],model_setting['hidden_size1']),volatile=volatile).cuda()
        hidden = autograd.Variable(torch.zeros(2, x.size()[0], model_setting['hidden_size1']), volatile=volatile)
        x,_ = self.GRU1(x,hidden)
        x = x[:,:,:model_setting['hidden_size1']] + x[:,:,model_setting['hidden_size1']:]
        # hidden = autograd.Variable(torch.zeros(2, x.size()[0],model_setting['hidden_size2']),volatile=volatile).cuda()
        hidden = autograd.Variable(torch.zeros(2, x.size()[0], model_setting['hidden_size2']), volatile=volatile)
        x,hn = self.GRU2(x,hidden)
        x = x[:, :, :model_setting['hidden_size2']] + x[:, :,model_setting['hidden_size2']:]          # n*200*size


        y2 = self.maxPool(x.transpose(1,2)).squeeze()
        y3 = self.avePool(x.transpose(1,2)).squeeze()

        # features = self.fc2(features)
        # features = nn.functional.tanh(features)
        y = torch.cat((y2,y3,features),1)

        # y = x.sum(dim=1)
        output = nn.functional.sigmoid(self.fc(y))
        return output

class DnnModle:

    def __init__(self,dim, embedding_matrix,alpha=3,trainable=True,loss = 'focalLoss'):
        super(DnnModle,self).__init__()

        # self.basenet = nn.DataParallel(baseNet(dim, embedding_matrix,trainable)).cuda()

        # self.basenet = baseNet(dim, embedding_matrix,trainable).cuda()
        self.basenet = baseNet(dim, embedding_matrix, trainable)

        self.optimizer = Nadam(
            [
            #     # {'params': self.basenet.module.GRU1.parameters()},
            #     # {'params': self.basenet.module.GRU2.parameters()},
            #     # {'params': self.basenet.module.fc.parameters()},
            #     # {'params': self.basenet.module.fc2.parameters()},
                {'params': self.basenet.GRU1.parameters()},
                {'params': self.basenet.GRU2.parameters()},
                {'params': self.basenet.fc.parameters()},
                # {'params': self.basenet.fc2.parameters()},
            #     # {'params':self.basenet.module.embedding.parameters() ,'lr':1e-5},
            ],
            lr=0.001,
        )
        self.basenet.train()

        if loss == 'focalLoss':
            self.loss_f = focallogloss(alpha=alpha)
        elif loss =='aucLoss':
            self.loss_f = auc_loss()
        elif loss =='ceLoss':
            self.loss_f = nn.BCELoss()

    def fit(self,X,features,Y):
        # comment = torch.autograd.Variable(X.cuda())
        # Y = torch.autograd.Variable(Y.cuda())
        # features = torch.autograd.Variable(features.cuda())
        # y_pred = self.basenet(comment,features,volatile=False)
        # loss = self.loss_f(y_pred,Y)
        # print(loss)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        comment = torch.autograd.Variable(X)
        Y = torch.autograd.Variable(Y)
        features = torch.autograd.Variable(features)
        y_pred = self.basenet(comment, features, volatile=False)
        loss = self.loss_f(y_pred, Y)
        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def predict(self,X,batchsize=1024):
        # self.basenet.eval()
        # comment = torch.autograd.Variable(torch.LongTensor(X['comment'].tolist()).cuda(), volatile=True)
        # features = torch.autograd.Variable(torch.FloatTensor(X['countFeature'].tolist()).cuda(),volatile=True)
        # y_pred = torch.zeros((len(comment),6)).cuda()
        # for i in range(0,len(comment),batchsize):
        #     y_pred[i:i+batchsize] = self.basenet(comment[i:i+batchsize],features[i:i+batchsize],volatile=True).data
        # self.basenet.train()
        # return y_pred.cpu().numpy()

        self.basenet.eval()
        comment = torch.autograd.Variable(torch.LongTensor(X['comment'].tolist()), volatile=True)
        features = torch.autograd.Variable(torch.FloatTensor(X['countFeature'].tolist()), volatile=True)
        y_pred = torch.zeros((len(comment), 6))
        for i in range(0, len(comment), batchsize):
            y_pred[i:i + batchsize] = self.basenet(comment[i:i + batchsize], features[i:i + batchsize], volatile=True).data
        self.basenet.train()
        return y_pred.numpy()





























