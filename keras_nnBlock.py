from keras.layers import CuDNNGRU,Input,Embedding,Bidirectional,Dropout,Dense,GlobalMaxPooling1D,GlobalAveragePooling1D,Concatenate,GRU,Multiply
from keras import backend as K
from keras.models import Model
from keras.optimizers import RMSprop,Nadam
from Ref_Data import NUM_TOPIC,USE_LETTERS,USE_TOPIC,model_setting

def focalLoss(y_true,y_pred,alpha=3):
    weight1 = K.pow(1 - y_pred, alpha)
    weight2 = K.pow(y_pred, alpha)
    loss = -(
            y_true * K.log(y_pred) * weight1 +
            (1 - y_true) * K.log(1 - y_pred) * weight2
    )
    loss = K.mean(loss,axis=-1) / 6
    return loss

def get_model(embedding_matrix ,trainable=False):

    comment_layer = Input(shape=(200,),name='comment')
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=trainable)(comment_layer)
    x = Bidirectional(GRU(model_setting['hidden_size1'], return_sequences=True,dropout=0.5),merge_mode='sum')(embedding_layer)
    x = Bidirectional(GRU(model_setting['hidden_size2'], return_sequences=True,dropout=0.5),merge_mode='sum')(x)
    x1 = GlobalMaxPooling1D()(x)
    x2 = GlobalAveragePooling1D()(x)

    dim_features = 4
    if USE_TOPIC:
        dim_features += NUM_TOPIC
    if USE_LETTERS:
        dim_features += 27
    features_layer = Input(shape=(dim_features,),name='countFeature')

    y = Concatenate()([x1,x2,features_layer])

    output_layer = Dense(6 , activation="sigmoid")(y)

    opt = Nadam(lr=0.0015)
    model = Model(inputs=[comment_layer,features_layer], outputs=output_layer)
    model.compile(loss=focalLoss,optimizer=opt)

    return model

































