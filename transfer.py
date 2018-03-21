import input
import nnBlock
import tool
import numpy as np
from Ref_Data import WEIGHT_FILE

def transfer(maxlen=200,modelname='rnn',loss="focalLoss"):
    setting= {
        'lr': 0.001,
        'decay': 0.004,
        'dropout': 0.3,
        # 'size1':170,    rnn
        # 'size2':80,
        'size1': 170,
        'size2': 80,
    }

    trainset,labels, embedding_matrix = \
        input.get_transfer_data(maxlen, fastText='crawl',
                        trainfile='clean_attack_annotated_comments.csv',target='attack')

    model = nnBlock.model(embedding_matrix,trainable=False,use_feature = False,
                      loss=loss,setting=setting)

    # layers = model.get_layer(modelname)
    layers = model.transfer_model(modelname)

    # generator = tool.Generate(trainset,labels, batchsize=100*256)


    for epoch in range(3):
        # samples_x, samples_y = generator.genrerate_samples()
        model.fit(trainset,labels, batch_size=256, epochs=1, verbose=1)

    for name,layer in layers.items():
        np.save(WEIGHT_FILE+loss+name+'_weight.npy',layer.get_weights())

if __name__=='__main__':
    transfer(200,'rnn','focalLoss')

