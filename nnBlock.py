from keras.layers import CuDNNGRU,Input,Embedding,Bidirectional,Dropout,Dense,GlobalMaxPooling1D,GlobalAveragePooling1D,Concatenate
from keras import backend as K
from keras.layers import Conv1D,Multiply,Permute,MaxPool1D
from keras.models import Model
from keras.optimizers import RMSprop,Nadam
from Ref_Data import NUM_TOPIC,USE_LETTERS,USE_TOPIC,model_setting,WEIGHT_FILE,BALANCE_GRAD
from keras import initializers
import numpy as np
from keras.engine.topology import Layer
K.clear_session()



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
        label_weight = K.softmax(self.kernel)
        if self.loss == 'focalLoss':
            weight1 = K.pow(1 - y_pred, 3)*label_weight
            weight2 = K.pow(y_pred, 3)*label_weight
            pos_y = K.log(y_pred)*weight1
            neg_y = K.log(1-y_pred)*weight2
        elif self.loss == "binary_crossentropy":  # celoss
            pos_y = K.log(y_pred) * label_weight
            neg_y = K.log(1 - y_pred) * label_weight
        else:
            raise NameError('loss name error')
        return K.concatenate([pos_y,neg_y], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def meanLoss(y_true,y_pred):
    y_true = y_true[:,:6]
    pos_y = y_pred[:,:6]
    neg_y = y_pred[:, 6:]
    loss = -(y_true * pos_y + (1 - y_true) *neg_y)
    loss = K.mean(loss,axis=-1)/6
    return loss

def focalLoss(y_true,y_pred,alpha=3):
    weight1 = K.pow(1 - y_pred, alpha)
    weight2 = K.pow(y_pred, alpha)
    loss = -(
            y_true * K.log(y_pred) * weight1 +
            (1 - y_true) * K.log(1 - y_pred) * weight2
    )
    loss = K.mean(loss,axis=-1) / 6
    return loss

def diceLoss(y_true, y_pred,smooth = 0):
    intersection = K.sum(y_true * y_pred)
    loss = - (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    return loss

class model:
    def __init__(self,embedding_matrix ,trainable=False,use_feature = True,
                 loss="binary_crossentropy",load_weight = False):

        self.comment_layer = Input(shape=(200,), name='comment')
        self.embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                    weights=[embedding_matrix], trainable=trainable)(self.comment_layer)

        self.use_feature = use_feature
        if use_feature:
            dim_features = 4
            if USE_TOPIC:
                dim_features += NUM_TOPIC
            if USE_LETTERS:
                dim_features += 27
            self.features_layer = Input(shape=(dim_features,), name='countFeature')

        self.load_weight = load_weight
        self.opt = Nadam(lr=0.001)

        self.lossname = loss
        if loss == 'focalLoss':
            self.loss = focalLoss
        elif loss == 'diceLoss':
            self.loss = diceLoss
        elif loss == 'binary_crossentropy':
            self.loss = 'binary_crossentropy'
        else:
            raise NameError("loss name error in model")

        if BALANCE_GRAD and loss!='diceloss':
            self.loss = meanLoss

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
            if self.use_feature:
                self.train_model = \
                    Model(inputs=[self.comment_layer, self.features_layer], outputs=output_layer)
            else:
                self.train_model = Model(inputs=[self.comment_layer], outputs=output_layer)
        else:
            self.train_model = self.result_model
        self.train_model.compile(optimizer=self.opt,loss=self.loss)

    def RNNmodel(self):

        gru1 = Bidirectional(CuDNNGRU(model_setting['hidden_size1'], return_sequences=True), merge_mode='ave')
        gru2 = Bidirectional(CuDNNGRU(model_setting['hidden_size2'], return_sequences=True), merge_mode='ave')

        layer = {
            'gru1':gru1,
            'gru2': gru2,
        }

        if self.load_weight:
            gru1_weight = np.load(WEIGHT_FILE+self.lossname+'gru1_weight.npy')
            gru1.set_weights(gru1_weight)
            gru2_weight = np.load(WEIGHT_FILE+self.lossname+'gru2_weight.npy')
            gru2.set_weights(gru2_weight)

        x = gru1(self.embedding_layer)
        x = Dropout(0.5)(x)
        x = gru2(x)
        x = Dropout(0.5)(x)
        x1 = GlobalMaxPooling1D()(x)
        x2 = GlobalAveragePooling1D()(x)
        if self.use_feature:
            y = Concatenate()([x1, x2, self.features_layer])
            output_layer = Dense(6, activation="sigmoid")(y)
            self.result_model = Model(inputs=[self.comment_layer, self.features_layer], outputs=output_layer)
        else:
            y = Concatenate()([x1, x2])
            output_layer = Dense(6, activation="sigmoid")(y)
            self.result_model = Model(inputs=[self.comment_layer], outputs=output_layer)

        self.set_loss(output_layer)

        return layer

    def CNNmodel(self):
        pass

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
        if self.use_feature:
            y = Concatenate()([x1, x2, self.features_layer])
            output_layer = Dense(6, activation="sigmoid")(y)
            self.result_model = Model(inputs=[self.comment_layer, self.features_layer], outputs=output_layer)
        else:
            y = Concatenate()([x1, x2])
            output_layer = Dense(6, activation="sigmoid")(y)
            self.result_model = Model(inputs=[self.comment_layer], outputs=output_layer)

        self.set_loss(output_layer)
        return layer


    def fit(self,X,Y,batch_size,epochs,verbose):
        if BALANCE_GRAD:
            self.train_model.fit(X,np.concatenate([Y,Y],axis=1),batch_size,epochs,verbose)
        else:
            self.train_model.fit(X,Y, batch_size, epochs, verbose)

    def predict(self,X,batch_size=2048, verbose=1):
        return self.result_model.predict(X,batch_size,verbose)

    def save_weights(self,path):
        self.train_model.save_weights(path)

    def load_weights(self,path,by_name=False):
        self.train_model.load_weights(path,by_name=by_name)



























