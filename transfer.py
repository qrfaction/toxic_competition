import input
import nnBlock
import tool
import numpy as np
from Ref_Data import WEIGHT_FILE

def transfer(maxlen=200,lang='nl',modelname='rnn',loss="focalLoss"):

    trainset,labels, embedding_matrix = \
        input.get_transfer_data(maxlen, language=lang,fastText='wiki.en.bin')

    model = nnBlock.model(embedding_matrix,trainable=False,use_feature = False,
                      loss=loss)

    layers = model.get_layer(modelname)

    generator = tool.Generate(trainset,labels, batchsize=100*256)


    for epoch in range(20):
        samples_x, samples_y = generator.genrerate_samples()
        model.fit(samples_x, samples_y, batch_size=256, epochs=1, verbose=1)

    for name,layer in layers.items():
        np.save(WEIGHT_FILE+loss+name+'_weight.npy',layer.get_weights())

if __name__=='__main__':
    transfer(200,'clean','rnn','focalLoss')

