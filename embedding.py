from fastText import load_model
from tqdm import tqdm
import numpy as np
import multiprocessing as mlp
import input
from nltk.tokenize import TweetTokenizer
from Ref_Data import PATH,FILTER_FREQ


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
        result = pool.apply_async(tokenize_worker,args=(sentences[i * aver_t:(i + 1) * aver_t],))
        results.append(result)
    pool.close()
    pool.join()

    tokenized_sentences = []
    for result in results:
        tokenized_sentences.extend(result.get())

    return tokenized_sentences

def tokenize_sentences(sentences,filter_word=True):

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
        words_dict = {}
        seq_list = []
        lenseq = {}
        for sentence in tqdm(sentences):
            seq = []
            for word in sentence:
                if filter_word and frequency[word]<= FILTER_FREQ:
                    continue
                if word not in words_dict:
                    words_dict[word] = len(words_dict)+1
                word_index = words_dict[word]
                seq.append(word_index)
            seq_list.append(seq)
            if len(seq) not in lenseq:
                lenseq[len(seq)]=0
            lenseq[len(seq)] +=1
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
    num_words = len(word_index) + 1
    # 停止符用0
    embedding_matrix = np.random.uniform(-0.25,0.25,size=(num_words,dimension))
    embedding_matrix[0] = 0

    ft_model = load_model(PATH + 'wiki.en.bin')
    print('num of word: ',len(word_index))
    if wordvecfile=='fasttext':
        for word, i in tqdm(word_index.items()):
            embedding_matrix[i] = ft_model.get_word_vector(word).astype('float32')
    else:
        embeddings_index = input.read_wordvec(wordvecfile)
        noword = {}
        num_noword = 0
        for word, i in tqdm(word_index.items()):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                noword[word]=frequency[word]
                num_noword+=1
                embedding_matrix[i] = ft_model.get_word_vector(word).astype('float32')
            else:
                embedding_matrix[i] = embedding_vector

        import json
        noword = sorted(noword.items(),key=lambda item:item[1],reverse=True)
        with open('test.json','w') as f:
            f.write(json.dumps(noword,indent=4, separators=(',', ': ')))
        print('miss:', num_noword)

    return embedding_matrix

