from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, Embedding
from keras.models import Model
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout,GRU,add
from keras.layers.pooling import MaxPool1D
from keras.optimizers import Adam
import prepocess
import input
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU
from keras.layers.merge import concatenate

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

    def DenseNet(input_layer, filters):
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
        DBlock4 = concatenate([DBlock4, DBlock3, DBlock2, DBlock1, input_layer])

        return DBlock4

    if name=='res_inception':
        return Res_Inception(input_layer,filters)
    elif name=='DenseNet':
        return DenseNet(input_layer,filters)

def get_model(num_words,EMBEDDING_DIM,embedding_matrix,maxlen,trainable=True):

    sequence_input = Input(shape=(maxlen,), dtype='int32')
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=maxlen,
                                trainable=trainable)(sequence_input)
    conv_layer=Conv1D(256,kernel_size=3,padding='same',strides=1)(embedding_layer)
    conv_layer = PReLU()(conv_layer)
    gru_layer = GRU(units=128, return_sequences=True, recurrent_dropout=0.2)(conv_layer)

    conv_layer =Conv1D(128, kernel_size=3, padding='same', strides=1)(gru_layer)
    conv_layer = PReLU()(conv_layer)
    gru_layer = GRU(units=64, return_sequences=True, recurrent_dropout=0.2)(conv_layer)

    conv_layer = Conv1D(64, kernel_size=3, padding='same', strides=2)(gru_layer)
    conv_layer = PReLU()(conv_layer)
    gru_layer = GRU(units=64, return_sequences=True,recurrent_dropout=0.2)(conv_layer)

    gru_layer = Bidirectional(GRU(units=32, return_sequences=False))(gru_layer)
    x = Dense(64)(gru_layer)
    x = PReLU()(x)
    x = Dropout(0.5)(x)
    output = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=[sequence_input], outputs=[output])
    optimizer = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def train(maxlen=100):

    train,test=input.read_dataset('clean_train.csv'),input.read_dataset('clean_test.csv')
    labels=input.read_dataset('labels.csv').values
    train, test,embedding_matrix=prepocess.comment_to_seq(train,test,maxlen=maxlen)

    model=get_model(len(embedding_matrix),300,embedding_matrix,maxlen=maxlen)

    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

    callbacks_list = [ early]  # early
    model.fit(train, labels, batch_size=128, epochs=1,callbacks=callbacks_list,
              verbose=True)

    sample_submission = input.read_dataset('sample_submission.csv')
    y_test = model.predict(test,verbose=1)
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    sample_submission[list_classes] = y_test

    sample_submission.to_csv("baseline.csv.gz", index=False,compression='gzip')

if __name__ == "__main__":
    train()