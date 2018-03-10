blend-of-blends-1/superblend_1.csv
lgb-gru-lr-lstm-nb-svm-average-ensemblesubmission.csv
bi-gru-cnn-poolings/submission.csv



pooled-gru-with-preprocessing/submission.csv
bi-gru-cnn-poolings/submission.csv')
pooled-gru-glove-with-preprocessing/submission.csv
toxic-avenger/submission.csv
toxicfile/sub9821.csv
toxic-glove/glove.csv
toxic-nbsvm/nbsvm.csv
toxic-hight-of-blending/hight_of_blending.csv
bidirectional-lstm-with-convolution/submission.csv



gru = pd.read_csv("../input/who09829/submission.csv")
gruglo = pd.read_csv("../input/pooled-gru-glove-with-preprocessing/submission.csv")
ave = pd.read_csv("../input/toxic-avenger/submission.csv")
s9821 = pd.read_csv("../input/toxicfile/sub9821.csv")
#glove = pd.read_csv('../input/toxic-glove/glove.csv')
#svm = pd.read_csv("../input/toxic-nbsvm/nbsvm.csv")
best = pd.read_csv('../input/toxic-hight-of-blending/hight_of_blending.csv')

