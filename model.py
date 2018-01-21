from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout,GRU
from keras.optimizers import Adam
import prepocess
import input
from keras.callbacks import EarlyStopping

def get_model(num_words,EMBEDDING_DIM,embedding_matrix,maxlen,trainable=True):

    sequence_input = Input(shape=(maxlen,), dtype='int32')
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=maxlen,
                                trainable=trainable)(sequence_input)
    
    gru_layer=Bidirectional(GRU(units=200,return_sequences=True))(embedding_layer)
    gru_layer = Bidirectional(GRU(units=128, return_sequences=True))(gru_layer)
    gru_layer = Bidirectional(GRU(units=64, return_sequences=True))(gru_layer)
    gru_layer = Bidirectional(GRU(units=32, return_sequences=False))(gru_layer)
    x = Dense(64, activation="relu")(gru_layer)
    x = Dropout(0.5)(x)
    output = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=[sequence_input], outputs=[output])
    optimizer = Adam(lr=0.02)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def train():
    train,test=input.read_dataset('clean_train.csv'),input.read_dataset('clean_test.csv')
    labels=input.read_dataset('labels.csv').values
    train, test,embedding_matrix=prepocess.comment_to_seq(train,test)

    model=get_model(len(embedding_matrix),300,embedding_matrix,1000)

    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

    callbacks_list = [ early]  # early
    model.fit(train, labels, batch_size=128, epochs=100, validation_split=0.25, callbacks=callbacks_list,
              verbose=True)

    sample_submission = input.read_dataset('sample_submission.csv')
    y_test = model.predict(test)
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    sample_submission[list_classes] = y_test

    sample_submission.to_csv("baseline.csv", index=False)

if __name__ == "__main__":
    train()