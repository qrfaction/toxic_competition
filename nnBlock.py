from keras.layers import CuDNNGRU,Input,Embedding,Bidirectional,Dropout,Dense,GlobalMaxPooling1D,GlobalAveragePooling1D,Concatenate
from keras.layers import Conv1D,Multiply,Permute,MaxPool1D,SpatialDropout1D,concatenate,Activation,BatchNormalization,add
from keras.models import Model
from keras.optimizers import RMSprop,Nadam
from Ref_Data import NUM_TOPIC,USE_LETTERS,USE_TOPIC
from Ref_Data import BATCHSIZE,WEIGHT_FILE,BALANCE_GRAD,USE_CHAR_VEC,LEN_CHAR_SEQ,USE_TFIDF,CHAR_N,LOG_DIR
import numpy as np
from keras.engine.topology import Layer
from keras import backend as K
from keras import initializers


K.clear_session()

USE_BOOST = True





class balanceGradLayer(Layer):

    def __init__(self,loss,**kwargs):
        self.output_dim = 12
        self.loss = loss
        super(balanceGradLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(6,),
                                      initializer=initializers.Ones(),
                                      trainable=True)
        super(balanceGradLayer, self).build(input_shape)

    def call(self, y_pred):
        # label_weight = K.softmax(self.kernel)
        label_weight = K.sigmoid(self.kernel)

        if self.loss == 'focalLoss':
            weight1 = K.pow(1 - y_pred,3)*label_weight
            weight2 = K.pow(y_pred, 3)*(1-label_weight)
            pos_y = K.log(y_pred)*weight1
            neg_y = K.log(1-y_pred)*weight2
        elif self.loss == "binary_crossentropy":  # celoss
            pos_y = K.log(y_pred) * label_weight
            neg_y = K.log(1 - y_pred) * (1-label_weight)
        else:
            raise NameError('loss name error')
        return K.concatenate([pos_y,neg_y], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def meanLoss(y_true,y_pred):
    y_true = y_true[:,:6]
    pos_y = y_pred[:,:6]
    neg_y = y_pred[:, 6:]
    loss = y_true * pos_y + (1 - y_true) *neg_y
    loss = -K.mean(loss,axis=-1)/2
    return loss

def focalLoss(y_true,y_pred):
    weight1 = K.pow(1 - y_pred, 1)
    weight2 = K.pow(y_pred, 1)
    loss = y_true * K.log(y_pred) * weight1 +\
        (1 - y_true) * K.log(1 - y_pred) * weight2

    loss = -K.mean(loss,axis=-1) / 6
    return loss

def diceLoss(y_true, y_pred,smooth = 0):
    intersection = K.sum(y_true * y_pred)
    loss = - (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    return loss

class boostLayer(Layer):

    def __init__(self,**kwargs):
        self.output_dim = 6
        super(boostLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(6,),
                                      initializer=initializers.Zeros(),
                                      trainable=True)
        super(boostLayer, self).build(input_shape)

    def call(self,scores):
        label_weight = K.sigmoid(self.kernel)
        last_scores = scores[:,:6]
        cur_scores = scores[:,6:]
        return cur_scores*label_weight + last_scores*(1-label_weight)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def pearsonLoss(y_true,y_pred):
    std_y_ture = K.std(y_true,axis=1)
    std_y_pred = K.std(y_pred, axis=1)
    mean_y_true = K.mean(y_true,axis=1)
    mean_y_pred = K.mean(y_pred, axis=1)
    y_true = (y_true-mean_y_true)/std_y_ture
    y_pred = (y_pred-mean_y_pred)/std_y_pred
    loss = K.batch_dot(y_true,y_pred, axes=1)
    loss = K.squeeze(loss,axis=0)
    return -K.mean(loss)

def rankLoss(y_true,y_pred):
    loss = 0
    for i in range(6):
        y = K.expand_dims(y_true[:,i],axis=1)    # [batch] -> [batch,1]
        y = K.repeat_elements(y,BATCHSIZE,1)          #  [batch,1] -> [batch,batch]
        y = y - K.transpose(y)

        x = K.expand_dims(y_pred[:, i], axis=1)  # [batch] -> [batch,1]
        x = K.repeat_elements(x,BATCHSIZE, 1)  # [batch,1] -> [batch,batch]
        x = x - K.transpose(x)

        label_y = K.pow(y,2)

        logloss = K.log(K.sigmoid(y * x))*label_y      # y = 0的不产生loss

        num_pos = K.sum(y_true[:,i])
        num_neg = K.sum(1-y_true[:,i])

        loss1 = -K.sum(logloss)/(num_neg*num_pos*2+1)

        loss2 = y_true[:,i] * K.log(y_pred[:,i]) + \
                    (1 - y_true[:,i]) * K.log(1 - y_pred[:,i])
        loss2 = -K.mean(loss2,axis=-1)
        loss +=loss1*loss2

    return loss/6


def char2vec(X,Y,embedding_matrix,batchsize = 102400,epochs=5):

    def get_char2vec_model(embedding_matrix,num_chars=5):
        char_layer = Input(shape=(num_chars,), name='char_layer')
        embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                            weights=[embedding_matrix], trainable=True)(char_layer)
        gru =CuDNNGRU(300, return_sequences=False)
        model = Model(inputs=[char_layer], outputs=gru)
        model.compile(optimizer=Nadam(0.001),loss=pearsonLoss)
        return model,embedding_layer

    model,embed_layer = get_char2vec_model(embedding_matrix)
    steps = len(X) // batchsize +1
    for epoch in range(epochs):
        for i in range(steps):
            sample_x = X[batchsize*i:batchsize*(i+1)]
            sample_y = Y[batchsize*i:batchsize*(i+1)]
            model.fit(sample_x,sample_y,verbose=1)
    np.save(WEIGHT_FILE+"char_vec.npy", embed_layer.get_weights())


class model:

    def __init__(self,embedding_matrix ,trainable=False,use_feature = True,
                 loss="binary_crossentropy",load_weight = False,char_weight=None,boost=False,
                 setting = None,maxlen=200):

        comment_layer = Input(shape=(maxlen,), name='comment')
        self.embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                    weights=[embedding_matrix], trainable=trainable)(comment_layer)

        self.cat_layers = []
        self.inputs = [comment_layer]

        self.select_feature(use_feature,char_weight)

        self.model_setting = setting

        self.load_weight = load_weight
        self.opt = Nadam(lr=setting['lr'],schedule_decay=setting['decay'])
        self.select_loss(loss)

        self.boost = boost
        if self.boost:
            self.boost_layer = Input(shape=(6,), name='boost')
            self.inputs.append(self.boost_layer)



    def select_loss(self,loss):
        self.lossname = loss
        print(loss)
        if loss == 'focalLoss':
            self.loss = focalLoss
        elif loss == 'diceLoss':
            self.loss = diceLoss
        elif loss == 'binary_crossentropy':
            self.loss = 'binary_crossentropy'
        elif loss == 'rankLoss':
            self.loss = rankLoss
        else:
            raise NameError("loss name error in model")

        if BALANCE_GRAD and loss != 'diceloss':
            self.loss = meanLoss

    def select_feature(self,use_feature,char_weight):
        if use_feature:
            dim_features = 7
            if USE_TOPIC:
                dim_features += NUM_TOPIC
            if USE_LETTERS:
                dim_features += 1
            if USE_TFIDF:
                dim_features += CHAR_N
            features_layer = Input(shape=(dim_features,), name='countFeature')
            self.inputs.append(features_layer)
            self.cat_layers.append(features_layer)

        if USE_CHAR_VEC:
            char_input = Input(shape=(LEN_CHAR_SEQ,), name='char')
            char_layer = Embedding(char_weight.shape[0],char_weight.shape[1],
                            weights=[char_weight], trainable=False)(char_input)
            gru3 = Bidirectional(CuDNNGRU(32,return_sequences=True), merge_mode='ave')(char_layer)
            x3 = GlobalMaxPooling1D()(gru3)
            x4 = GlobalAveragePooling1D()(gru3)
            self.cat_layers +=[x3,x4]
            self.inputs.append(char_input)


    def get_layer(self,modelname):
        if modelname == 'rnn':
            return self.RNNmodel()
        elif modelname == 'cnn':
            return self.CNNmodel()
        elif modelname == 'cnnGLU':
            return self.CNNGLU()

    def set_loss(self,output_layer):
        if BALANCE_GRAD:
            output_layer = balanceGradLayer(self.lossname)(output_layer)
            self.train_model=Model(inputs=self.inputs, outputs=output_layer)
        else:
            self.train_model = self.result_model
        self.train_model.compile(optimizer=self.opt,loss=self.loss)

    def RNNmodel(self):

        size_1 = self.model_setting['size1']
        size_2 = self.model_setting['size2']
        p = self.model_setting['dropout']
        gru1 = Bidirectional(CuDNNGRU(size_1, return_sequences=True), merge_mode='ave')
        gru2 = Bidirectional(CuDNNGRU(size_2, return_sequences=True), merge_mode='ave')

        layer = {
            'gru1':gru1,
            'gru2':gru2,
        }

        embedding_layer = SpatialDropout1D(p)(self.embedding_layer)

        x = gru1(embedding_layer)

        x = gru2(x)        # 200 * 80

        x1 = GlobalMaxPooling1D()(x)
        x2 = GlobalAveragePooling1D()(x)

        self.cat_layers += [x1,x2]

        y = Concatenate()(self.cat_layers)
        # y = Dense(90,activation='relu')(y)

        fc = Dense(6)(y)
        result_layer = Activation(activation='sigmoid')(fc)


        if self.boost:
            result_layer = concatenate([result_layer,self.boost_layer],axis=-1)
            result_layer = boostLayer()(result_layer)

        self.result_model = Model(inputs=self.inputs, outputs=result_layer)

        print(self.result_model.summary())

        if self.lossname == 'rankLoss':
            loss_layer = fc
        else:
            loss_layer = result_layer
        self.set_loss(loss_layer)

        if self.load_weight:
            name = self.lossname
            name = 'focalLoss'
            gru1_weight = np.load(WEIGHT_FILE+name+'gru1_weight.npy')
            gru1.set_weights(gru1_weight)
            gru2_weight = np.load(WEIGHT_FILE+name+'gru2_weight.npy')
            gru2.set_weights(gru2_weight)

        return layer

    def CNNmodel(self):

        p = self.model_setting['dropout']
        size_1 = self.model_setting['size1']
        size_2 = self.model_setting['size2']

        embedding_layer = SpatialDropout1D(p)(self.embedding_layer)
        # embedding_layer = self.embedding_layer


        conv2 = Conv1D(size_1,2, padding='same',activation='relu')(embedding_layer)
        conv2 = MaxPool1D(pool_size=3,strides=2,padding='same')(conv2)
        conv2 = Conv1D(size_2,2, padding='same',activation='relu')(conv2)
        conv2 = GlobalMaxPooling1D()(conv2)

        conv3 = Conv1D(size_1, 3, padding='same',activation='relu')(embedding_layer)
        conv3 = MaxPool1D(pool_size=3,strides=2,padding='same')(conv3)
        conv3 = Conv1D(size_2, 3, padding='same',activation='relu')(conv3)
        conv3 = GlobalMaxPooling1D()(conv3)

        conv4 = Conv1D(size_1, 4, padding='same',activation='relu')(embedding_layer)
        conv4 = MaxPool1D(pool_size=3, strides=2, padding='same')(conv4)
        conv4 = Conv1D(size_2, 4, padding='same',activation='relu')(conv4)
        conv4 = GlobalMaxPooling1D()(conv4)

        conv5 = Conv1D(size_1, 5, padding='same',activation='relu')(embedding_layer)
        conv5 = MaxPool1D(pool_size=3,strides=2,padding='same')(conv5)
        conv5 = Conv1D(size_2, 5, padding='same',activation='relu')(conv5)
        conv5 = GlobalMaxPooling1D()(conv5)


        # add1 = add([conv2,conv3])
        # add2 = add([conv4, conv5])

        self.cat_layers+=[
            conv2,conv3,conv5,conv4,
            # add1,add2
        ]

        # 拼接三个模块
        y = Concatenate()(self.cat_layers)
        # y = Dense(128,activation='selu')(y)
        fc = Dense(6)(y)
        result_layer = Activation(activation='sigmoid')(fc)

        if self.boost:
            result_layer = concatenate([result_layer, self.boost_layer], axis=-1)
            result_layer = boostLayer()(result_layer)

        self.result_model = Model(inputs=self.inputs, outputs=result_layer)

        print(self.result_model.summary())

        if self.lossname == 'rankLoss':
            loss_layer = fc
        else:
            loss_layer = result_layer
        self.set_loss(loss_layer)

        return {}

    def transfer_model(self,modelname):
        layers = self.get_layer(modelname)
        fc =Concatenate()(self.cat_layers)
        fc = Dense(1)(fc)

        self.result_model = Model(inputs=self.inputs, outputs=fc)
        self.train_model = self.result_model
        self.train_model.compile(optimizer=self.opt,loss='mse')
        return layers

    def CNNGLU(self):
        layer = {}
        def GLU(x,dim,num):
            conv1 = Conv1D(filters=dim, kernel_size=1, padding='same')
            conv2 = Conv1D(filters=dim, kernel_size=1, padding='same', activation='sigmoid')
            layer['glu_conv1_'+str(num)] = conv1
            layer['glu_conv2_'+str(num)] = conv2
            if self.load_weight:
                conv1_name = WEIGHT_FILE + self.lossname + 'glu_conv1_'+str(num)+'_weight.npy'
                conv2_name = WEIGHT_FILE + self.lossname + 'glu_conv2_'+str(num)+'_weight.npy'
                conv1_weight = np.load(conv1_name)
                conv1.set_weights(conv1_weight)
                conv2_weight = np.load(conv2_name)
                conv2.set_weights(conv2_weight)
            x1 = conv1(x)
            x2 = conv2(x)
            return Multiply()([x1,x2])

        x = Permute((2, 1))(self.embedding_layer)
        x = GLU(x,400,1)
        x = MaxPool1D(strides=2,padding='same')(x)
        x = GLU(x,300,2)
        x = GLU(x,200,3)
        x = GLU(x,100,4)
        x1 = GlobalMaxPooling1D()(x)
        x2 = GlobalAveragePooling1D()(x)

        self.cat_layers +=[x1,x2]

        y = Concatenate()(self.cat_layers)
        output_layer = Dense(6, activation="sigmoid",)(y)
        self.result_model = Model(inputs=self.inputs, outputs=output_layer)
        self.set_loss(output_layer)
        return layer

    def fit(self,X,Y,epochs,batch_size=None,sample_weight=None,verbose=0):
        if BALANCE_GRAD:
            self.train_model.fit(X,np.concatenate([Y,Y],axis=1),batch_size,
                                 epochs,verbose,sample_weight=sample_weight)
        else:
            self.train_model.fit(X,Y, batch_size, epochs, verbose,
                                 sample_weight=sample_weight)

    def predict(self,X,batch_size=2048, verbose=0):
        return self.result_model.predict(X,batch_size,verbose)

    def evaulate(self,X,Y,batch_size=2048,sample_weight=None,verbose=0):
        return self.train_model.evaluate(X, Y, batch_size,verbose, sample_weight)

    def save_weights(self,path):
        self.train_model.save_weights(path)

    def load_weights(self,path,by_name=False):
        self.train_model.load_weights(path,by_name=by_name)

    def recompile(self,loss=None):

        if loss is None:
            self.loss = self.loss
        else:
            self.select_loss(loss)
        self.train_model.compile(optimizer=self.opt,loss=self.loss)













