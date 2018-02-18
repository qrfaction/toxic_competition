import prepocess
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Ref_Data import PATH

def Sanitize():
    def JoinAndSanitize(cmt, annot):
        df = cmt.set_index('rev_id').join(annot.groupby(['rev_id']).mean())
        return df

    usecol = ['rev_id', 'quoting_attack', 'recipient_attack', 'third_party_attack', 'other_attack', 'attack']
    attack_annot = pd.read_table(PATH + 'attack_annotations.tsv', usecols=usecol)
    usecol = ['rev_id', 'comment_text']
    attack_cmt = pd.read_table(PATH + 'attack_annotated_comments.tsv', usecols=usecol)

    usecol = ['rev_id', 'comment_text']
    toxic_cmt = pd.read_table(PATH + 'toxicity_annotated_comments.tsv', usecols=usecol)

    usecol = ['rev_id', 'toxicity_score', 'toxicity']
    toxic_annot = pd.read_table(PATH + 'toxicity_annotations.tsv', usecols=usecol)

    toxic_cmt = JoinAndSanitize(toxic_cmt,toxic_annot)
    attack_cmt = JoinAndSanitize(attack_cmt,attack_annot)

    prepocess.clean_dataset(attack_cmt,'clean_attack_annotated_comments.csv')
    prepocess.clean_dataset(toxic_cmt,'clean_toxicity_annotated_comments.csv')




def Tfidfize(df):

    max_vocab = 200000

    comment = 'comment' if 'comment' in df else 'comment_text'

    tfidfer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_vocab,
                              use_idf=1, stop_words='english',
                              smooth_idf=1, sublinear_tf=1)
    tfidf = tfidfer.fit_transform(df[comment])

    return tfidf, tfidfer

def get_label_feat(trainfile,target):
    from Ref_Data import replace_word
    dataset = pd.read_csv(PATH+trainfile)
    dataset.dropna(inplace=True)
    print(dataset.shape)

    X, tfidfer = Tfidfize(dataset)
    Y = dataset[target].values

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    ridge = Ridge()
    mse_score = -cross_val_score(ridge, X, Y, scoring='neg_mean_squared_error')
    print(mse_score.mean())

    model = ridge.fit(X,Y)

    train_orig = pd.read_csv('data/clean_train.csv')
    test_orig = pd.read_csv('data/clean_test.csv')
    train_orig['comment_text']=train_orig['comment_text'].fillna(replace_word['unknow'])
    test_orig['comment_text'] = test_orig['comment_text'].fillna(replace_word['unknow'])

    tfidf_train = tfidfer.transform(train_orig['comment_text'])
    tfidf_test = tfidfer.transform(test_orig['comment_text'])
    train_scores = model.predict(tfidf_train)
    test_scores = model.predict(tfidf_test)

    return train_scores,test_scores

def get_label_feature():
    import multiprocessing as mlp

    Sanitize()

    results = []
    features = {
        'attack':'clean_attack_annotated_comments.csv',
        'quoting_attack':'clean_attack_annotated_comments.csv',
        'recipient_attack':'clean_attack_annotated_comments.csv',
        'third_party_attack':'clean_attack_annotated_comments.csv',
        'other_attack':'clean_attack_annotated_comments.csv',
        'toxicity':'clean_toxicity_annotated_comments.csv',
        'toxicity_score':'clean_toxicity_annotated_comments.csv'
    }
    pool = mlp.Pool(mlp.cpu_count())
    for feat,file in features.items():
        result = pool.apply_async(get_label_feature, args=(file,feat))
        results.append(result)
    pool.close()
    pool.join()

    train_orig = pd.read_csv('data/clean_train.csv')
    test_orig = pd.read_csv('data/clean_test.csv')
    for i,(feat, file) in enumerate(features.items()):
        train_feat,test_feat = results[i].get()
        train_orig[feat+'level'] = train_feat
        test_orig[feat+'level'] = test_feat
    train_orig.to_csv(PATH + 'clean_train.csv', index=False)
    test_orig.to_csv(PATH + 'clean_test.csv', index=False)

if __name__ == '__main__':
    get_label_feature()






