import input
import nnBlock
import tool
import numpy as np
from RefData import WEIGHT_FILE

def transfer(maxlen=200,lang='nl',modelname='rnn',loss="focalLoss"):
    trainfile = lang+'_train.csv'

    trainset,labels, embedding_matrix = \
        input.get_transfer_data(maxlen, trainfile=trainfile, language=lang)

    getmodel = nnBlock.model(embedding_matrix,trainable=False,use_feature = False,
                      loss=loss)

    model,layers = getmodel.get_model(modelname)

    generator = tool.Generate(trainset,labels, batchsize=100*256)


    for epoch in range(30):
        samples_x, samples_y = generator.genrerate_samples()
        model.fit(samples_x, samples_y, batch_size=256, epochs=1, verbose=1)

    for name,layer in layers.items():
        np.save(WEIGHT_FILE+loss+name+'_weight.npy',layer.get_weights())

if __name__=='__main__':
    transfer(200,'nl','rnn','focalLoss')

