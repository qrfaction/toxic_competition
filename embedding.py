from nltk.tokenize import word_tokenize
from tqdm import tqdm
import numpy as np
import multiprocessing as mlp
import input


def tokenize_worker(sentences):
    sentences = [ word_tokenize(seq) for seq in tqdm(sentences)]
    return sentences

def tokenize_sentences(sentences):

    def step_tokenize(sentences):
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

    def step_to_seq(sentences):
        " 句子转序列 "
        words_dict = {}
        seq_list = []
        for sentence in tqdm(sentences):
            seq = []
            for word in sentence:
                if word not in words_dict:
                    words_dict[word] = len(words_dict)+1
                word_index = words_dict[word]
                seq.append(word_index)
            seq_list.append(seq)
        return seq_list,words_dict

    sentences = step_tokenize(sentences)
    return step_to_seq(sentences)

def get_embedding_matrix(sentences,maxlen,dimension,wordvecfile):
    from keras.preprocessing.sequence import pad_sequences
    print('tokenize word')
    sentences , word_index = tokenize_sentences(sentences)
    sentences = pad_sequences(sentences, maxlen=maxlen, truncating='post')
    sentences = np.array(sentences)

    embeddings_index = input.read_wordvec(wordvecfile)

    print('get embedding matrix')
    num_words = len(word_index) + 1
    # 未知单词用0  停止符用-1
    embedding_matrix = np.zeros((num_words,dimension))
    embeddings_index[0] = -1
    noword = 0
    for word, i in tqdm(word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is None:
            noword += 1
        else:
            embedding_matrix[i] = embedding_vector
    print('miss:', noword)

    return sentences,embedding_matrix

