import time
import warnings
import re
from tqdm import tqdm
from Ref_Data import APPO
from Ref_Data import replace_word
import input
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import createFeature

warnings.filterwarnings("ignore")
PATH='data/'


eng_stopwords = set(stopwords.words("english"))
lem = WordNetLemmatizer()

def cleanComment(comments):
    """
    This function receives comments and returns clean word-list
    """
    def correct_typos(comment):
        comment = re.sub('(f u c k)', ' fuck ', comment)
        comment = re.sub('( f you)',' fuck you ',comment)
        comment = re.sub('(f u c k e r)', ' fucker ', comment)
        comment = re.sub('( ass monkey)', ' asshole ', comment)
        comment = re.sub('( a s s)',' ass ',comment)
        comment = re.sub('(a$$)', 'ass', comment)
        comment = re.sub('( w t f)',' wtf ',comment)
        comment = re.sub('( s t f u)',' stfu ' ,comment)
        pattern = '(motha fuker)|(motha fucker)|(motha fukkah)|(motha fukker)|(mother fucker)|(mother fukah)|(mother fuker)|(mother fukkah)|(mother fukker)|'
        pattern +='(mutha fucker)|(mutha fukah)|(mutha fuker)|(mutha fukkah)|(mutha fukker)'
        comment = re.sub(pattern,' motherfucker ',comment)
        comment = re.sub('( b!\+ch)|( b!tch)|( bi\+ch)',' bitch ',comment)
        comment = re.sub('( s\.o\.b\.)|( s\.o\.b)',' sob ',comment)
        comment = re.sub('( sh!t)|( shi\+)|( sh!\+)',' shit ',comment)
        comment = re.sub('( blow job)','blowjob',comment)
        comment = re.sub("( let's )",' let us ',comment)
        comment = re.sub("('s )", ' is ', comment)
        comment = re.sub("(fack you)|(fack u)",' fuck you ',comment)
        comment = re.sub('(go fack)','go fuck',comment)
        comment = re.sub('( \d\d:\d\d)',replace_word['num'],comment)
        comment = re.sub('( anti-)', ' anti ', comment)
        return comment

    patternLink = '(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
    patternIP = '\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}'
    patternEmail = '[A-Za-z\d]+([-_.][A-Za-z\d]+)*@([A-Za-z\d]+[-.])+[A-Za-z\d]{2,4}'

    from nltk.tokenize import TweetTokenizer
    tknzr = TweetTokenizer()

    clean_comments = []

    for comment in tqdm(comments):

        comment = comment.lower()
        # 去除邮箱    邮箱先去 再去IP


        comment = re.sub(patternEmail, ' ', comment)
        # 去除IP
        comment = re.sub(patternIP, " ", comment)
        # 去除usernames
        comment = re.sub("\[\[.*\]", " ", comment)
        # 去除网址
        comment = re.sub(patternLink, " ", comment)
        comment = correct_typos(comment)
        # 去除非ascii字符
        comment = re.sub("[^\x00-\x7F]+", " ", comment)
        comment = re.sub("(\d+\.\d+)",replace_word['num'],comment)
        comment = re.sub("\.+", ' . ', comment)            #帮助分词
        comment = re.sub('[\|=\*/\`\~\\\\\}\{]+', ' ', comment)
        comment = re.sub('[\"]+', ' " ', comment)
        comment = re.sub('\'{2,}', ' " ', comment)

        # 分词
        words = tknzr.tokenize(comment)

        # 拼写纠正 以及 you're -> you are
        words = [APPO[word] if word in APPO else word for word in words]

        # 提取词干
        words = [lem.lemmatize(word, "v") for word in words]

        # 数字统一
        for i in range(len(words)):
            if words[i].isdigit() and words[i]!='911':
                words[i] = replace_word['num']

        # words = [w for w in words if w not in eng_stopwords]
        comment = " ".join(words)

        comment = comment.lower()
        comment = re.sub('\s+',' ',comment)
        comment = re.sub('(\. )+',' . ',comment)
        comment = re.sub('(\. \.)+',' . ',comment)
        comment = re.sub('("")+', ' ', comment)


        # 纠正拼写错误/
        # for word,pos in tknzr(comment):
        #     if w_dict.check(word) == False:
        #         try:
        #             comment = comment[:pos] + \
        #                       w_dict.suggest(word)[0] + \
        #                       comment[pos+len(word):]
        #             print(word,w_dict.suggest(word)[0])
        #         except IndexError:
        #             continue
        clean_comments.append(comment)
    return clean_comments

def clean_dataset(dataset,filename):
    import multiprocessing as mlp

    results = []
    pool = mlp.Pool(mlp.cpu_count())

    comments = list(dataset['comment_text'])
    aver_t = int(len(dataset) / mlp.cpu_count()) + 1
    for i in range(mlp.cpu_count()):
        result = pool.apply_async(cleanComment, args=(comments[i * aver_t:(i + 1) * aver_t],))
        results.append(result)
    pool.close()
    pool.join()

    clean_comments = []
    for result in results:
        clean_comments.extend(result.get())

    dataset['comment_text'] = clean_comments

    dataset.to_csv(PATH+filename,index=False)



def splitTarget(filename):
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    labels=input.read_dataset(filename,list_classes)
    labels.to_csv(PATH+'labels.csv',index=False)


def pipeline(
        file =('train.csv','test.csv','train_fr.csv','train_es.csv','train_de.csv')
    ):
    for filename in tqdm(file):
        dataset = input.read_dataset(filename)
        dataset.fillna(replace_word['unknow'],inplace=True)
        dataset = createFeature.countFeature(dataset)
        clean_dataset(dataset,'clean_'+filename)



if __name__ == "__main__":
    pipeline()
