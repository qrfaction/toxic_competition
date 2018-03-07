from fastText import load_model
from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing as mlp
import input
from nltk.tokenize import TweetTokenizer
from Ref_Data import PATH,FILTER_FREQ,USE_POSTAG,POSTAG_DIM


def tokenize_worker(sentences):
    tknzr = TweetTokenizer()
    sentences = [tknzr.tokenize(seq) for seq in tqdm(sentences)]
    return sentences

def tokenize_word(sentences):
    " 多进程分词"
    results = []
    pool = mlp.Pool(mlp.cpu_count())
    aver_t = int(len(sentences)/ mlp.cpu_count()) + 1
    for i in range(mlp.cpu_count()):
        result = pool.apply_async(tokenize_worker,
                                  args=(sentences[i * aver_t:(i + 1) * aver_t],))
        results.append(result)
    pool.close()
    pool.join()

    tokenized_sentences = []
    for result in results:
        tokenized_sentences.extend(result.get())

    return tokenized_sentences

def tokenize_sentences(sentences):

    def step_cal_frequency(sentences):
        frequency = {}
        for sentence in tqdm(sentences):
            for word in sentence:
                if frequency.get(word) is None:
                    frequency[word] = 0
                frequency[word]+=1
        import json
        all_word_frequency = sorted(frequency.items(), key=lambda item: item[1], reverse=True)
        with open('allWordFrequency.json', 'w') as f:
            f.write(json.dumps(all_word_frequency, indent=4, separators=(',', ': ')))
        return frequency

    def step_to_seq(sentences,frequency):
        " 句子转序列 "
        words_dict = { }
        seq_list = []

        for sentence in tqdm(sentences):
            seq = []
            for word in sentence:
                if frequency[word]<= FILTER_FREQ :
                    continue
                if word not in words_dict:
                    words_dict[word] = len(words_dict) + 1
                word_index = words_dict[word]
                seq.append(word_index)
            seq_list.append(seq)
        return seq_list,words_dict,frequency

    sentences = tokenize_word(sentences)
    freq = step_cal_frequency(sentences)
    return step_to_seq(sentences,freq)

def get_wordvec(word_index,frequency,wordvecfiles):
    vec_matrix = {}
    for file,dimension in wordvecfiles:
        vec_matrix[file] = get_embedding_matrix(word_index,frequency,dimension,file)
    return vec_matrix

def get_embedding_matrix(word_index,frequency,dimension,wordvecfile):
    print('get embedding matrix')
    if USE_POSTAG:
        dimension +=POSTAG_DIM
        postag_vec = pd.read_csv(PATH+'postagVec.csv')
        import json
        with open(PATH+"word2tag.json",'r') as f:
            word2tag = json.loads(f.read())

    num_words = len(word_index) + 1
    # 停止符用0
    embedding_matrix = np.zeros((num_words,dimension))
    print(embedding_matrix.shape)
    if USE_POSTAG:
        dimension -=POSTAG_DIM
        for word, i in tqdm(word_index.items()):
            tag = word2tag[word]
            embedding_matrix[i,dimension:] = np.array(postag_vec[tag])

    ft_model = load_model(PATH + 'wiki.en.bin')
    print('num of word: ',len(word_index))
    for word, i in tqdm(word_index.items()):
        embedding_matrix[i, :dimension] = ft_model.get_word_vector(word).astype('float32')
    del ft_model

    if wordvecfile !='fasttext':
        embeddings_index = input.read_wordvec(wordvecfile)
        noword = {}
        num_noword = 0
        for word, i in tqdm(word_index.items()):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                noword[word]=frequency[word]
                num_noword+=1
            else:
                embedding_matrix[i,:dimension] = embedding_vector

        import json
        noword = sorted(noword.items(),key=lambda item:item[1],reverse=True)
        with open('test.json','w') as f:
            f.write(json.dumps(noword,indent=4, separators=(',', ': ')))
        print('miss:', num_noword)

    return embedding_matrix

def char_analyzer(word):
    if len(word)<3:
        return [word]
    else:
        return [word[i: i + 3]  for i in range(len(word) - 2)]

def batch_char_analyzer(sentences,to_string=True):
    ch_seqs = []
    for sent in sentences:
        seq = []
        for word in sent:
            seq.extend(char_analyzer(word))
        if to_string:
            seq=' '.join(seq)
        ch_seqs.append(seq)
    return ch_seqs

















