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
import json

warnings.filterwarnings("ignore")
PATH='data/'

def cleanComment(comments):
    """
    This function receives comments and returns clean word-list
    """
    def correct_typos(comment):
        pattern = '(motha fuker)|(motha fucker)|(motha fukkah)|(motha fukker)|(mother fuckers{0,})|(mother fukah)|(mother fuker)|(mother fukkah)|(mother fukker)|'
        pattern += '(mutha fucker)|(mutha fukah)|(mutha fuker)|(mutha fukkah)|(mutha fukker)|(mu\,the\,rfu\,ckers)|(mothjer fucker)|(motha fuckas{0,})'
        comment = re.sub(pattern, ' motherfucker ', comment)
        comment = re.sub('( f off)|(f u c k o f f)', ' fuck off ', comment)
        comment = re.sub('(f u c k t a r d)', ' fucktard ', comment)
        comment = re.sub('(go f yourself)', ' go fuck yourself ', comment)
        comment = re.sub('(f u c k e r)|(fu\*er)', ' fucker ', comment)
        comment = re.sub('( fuc king)|( fuc ing)|(f u c k i n g)|(fuck!ng)|(f\. u\. c\. k\. i\. n\. g)|(fucking+)', ' fucking ', comment)
        comment = re.sub('(f u c k)|( fuc k)|(f uc k)|(fu\.ck)|(f\*ck)|(fuck{3,})|(fffffffff   uuuuuu     uuuuu   ccccccccccccc  kkkkk)|(f\. u\. c\. k)|(fu\,ck )', ' fuck ', comment)
        comment = re.sub('(fuck){2,}', ' fuck fuck fuck fuck ', comment)
        comment = re.sub('(blahblahman\d+)', ' blah blah man ', comment)
        comment = re.sub('( f you)|(fack you)|(fack u)',' fuck you ',comment)
        comment = re.sub('(b i t c h)|(b!tch)', 'bitch', comment)
        comment = re.sub('(c u n t s)|(c\,un\,t)', ' cunt ', comment)
        comment = re.sub('(c u n t)', ' cunt ', comment)
        comment = re.sub('(d a m e)', ' dame ', comment)
        comment = re.sub('(w\,hor\,es)', ' whore ', comment)
        comment = re.sub('(idi\.o\.t)|(id\.iot)', ' idiot ', comment)
        comment = re.sub('(st\.u\.p\.id)|(s\'tu\.pi\.d)', ' stupid ', comment)

        comment = re.sub('(w a n k e r)', ' wanker ', comment)
        comment = re.sub('(d i c k h e a d)', ' dickhead ', comment)
        comment = re.sub('(d e m o n s)', ' demon ', comment)
        comment = re.sub('(lov3r)', ' lover ', comment)
        comment = re.sub('(f@ggot)|(fagg0t)|(fa ggot)|(f\.a\.g\.g\.o\.t)|(f a g g o t)|(fa\,gg\,ot)', ' faggot ', comment)
        comment = re.sub('(b u m s)', ' bum ', comment)
        comment = re.sub('( c ock)|(c\*ck)', ' cock ', comment)
        comment = re.sub('(a r m p i t s)', ' armpit ', comment)
        comment = re.sub('(s m e l l)', ' smell ', comment)
        comment = re.sub('(s  t  i  n  k  y)', ' stinky ', comment)
        comment = re.sub('(s u ck)|(s u c k)|($uck)|(su ck )|( suck{3,})', ' suck ', comment)
        comment = re.sub('(d i c k)|(d!ck)|( di ck )', ' dick ', comment)
        comment = re.sub('(pen!s)', ' penis ', comment)
        comment = re.sub('( ass monkey)', ' asshole ', comment)
        comment = re.sub('( p i s s)|(p\.i\.s\.s)', ' piss ', comment)
        comment = re.sub('( h e l l)', ' hell ', comment)
        comment = re.sub('( a s s)|(a\$\$)( a s  s )',' ass ',comment)
        comment = re.sub('( w t f)',' wtf ',comment)
        comment = re.sub('(k!kes{0,})', ' kike ', comment)
        comment = re.sub('(hijos de puta)', ' son of bitch ', comment)
        comment = re.sub('( s t f u)',' stfu ' ,comment)
        comment = re.sub('(n i g g e r)|(nig gger)|(n e g r o)|(n!ger)|(n!gger)|(n!gga)|(n!gg@r)','nigger',comment)

        comment = re.sub('( b!\+ch)|( b!tch)|( bi\+ch)|( b!t\*h)|(b\,itch\,es)|(bi\,tc\,h)',' bitch ',comment)
        comment = re.sub('( s\.o\.b\.)|( s\.o\.b)',' sob ',comment)
        comment = re.sub('( sh!t)|( shi\+)|( sh!\+)|( shi t )',' shit ',comment)
        comment = re.sub('( p ussy)', ' pussy ', comment)
        comment = re.sub('( let\'s )', ' let us ', comment)
        comment = re.sub('(\'s )', ' ', comment)
        comment = re.sub('(blow jobs)|(blowjobs)|(blow job)', ' blowjob ', comment)

        comment = re.sub('(go fack)','go fuck',comment)
        comment = re.sub('( \d\d:\d\d)',replace_word['num'],comment)
        comment = re.sub('@', 'a', comment)
        comment = re.sub('\$', 's', comment)

        return comment

    patternLink = '(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
    patternIP = '\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    patternEmail = '[A-Za-z\d]+([-_.][A-Za-z\d]+)*@([A-Za-z\d]+[-.])+[A-Za-z\d]{2,4}'
    patternNum = '(\d+\.\d+)|(\d+\,)|(\d+\-)|(\d+\:)|(\$\d+)|(\$\d+\.\d+)'
    patternRgb = '#(?:[0-9a-f]{3}){1,2}'
    from nltk.tokenize import TweetTokenizer
    tknzr = TweetTokenizer()
    lem = WordNetLemmatizer()
    clean_comments = []

    for comment in tqdm(comments):

        comment = comment.lower()
        # 去除邮箱    邮箱先去 再去IP

        comment = re.sub(patternEmail, replace_word['link'], comment)
        # 去除IP
        comment = re.sub(patternIP, replace_word['link'], comment)
        # 去除usernames
        comment = re.sub("\[\[.*\]", " ", comment)
        # 去除网址
        comment = re.sub(patternLink, replace_word['link'], comment)

        comment = re.sub(patternRgb, " ", comment)

        comment = correct_typos(comment)
        # 去除非ascii字符
        comment = re.sub("[^\x00-\x7F]+", " ", comment)

        comment = re.sub(patternNum,replace_word['num'],comment)
        comment = re.sub("\.+", ' . ', comment)            #帮助分词
        comment = re.sub('[\|=\*/\`\~\\\\\}\{]+', ' ', comment)
        comment = re.sub('[\"]+', ' " ', comment)
        comment = re.sub('\'{2,}', ' " ', comment)

        # 分词
        words = tknzr.tokenize(comment)

        words = [APPO[word] if word in APPO else word for word in words]

        # 数字统一
        for i in range(len(words)):
            if words[i].isdigit() and words[i]!='911':
                words[i] = replace_word['num']

        comment = " ".join(words)

        comment = comment.lower()
        comment = re.sub("-", ' ', comment)
        comment = re.sub('\s+',' ',comment)
        comment = re.sub('(\. )+',' . ',comment)
        comment = re.sub('(\. \.)+',' . ',comment)
        comment = re.sub('("")+', '', comment)
        comment = re.sub('( \' )', ' " ', comment)
        comment = re.sub('(\( \))+', '', comment)

        # 分词
        words = tknzr.tokenize(comment)
        words = [APPO[word] if word in APPO else word for word in words]

        comment = " ".join(words)
        # for i in range(97,97+26):
        #     ch = chr(i)
        #     comment = re.sub(ch +'{3,}',ch , comment)

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

def translation_sub(dataset,file):
    # 将训练测试集中的外国语言替换成翻译后的
    with open('translation.json','r') as f:
        translation = json.loads(f.read())
    for key,value in tqdm(translation.items()):
        if key[:2] == file:
            index = int(key[2:])
            dataset.loc[index,'comment_text'] = value[0]
    return dataset


def pipeline(
    file = ( 'train.csv','test.csv',
            # 'train_fr.csv','train_es.csv','train_de.csv'
            )
    ):
    for filename in tqdm(file):
        dataset = input.read_dataset(filename)
        dataset = translation_sub(dataset,filename[:2])
        dataset.fillna(replace_word['unknow'],inplace=True)
        dataset = createFeature.countFeature(dataset)
        clean_dataset(dataset,'clean_'+filename)

    createFeature.get_char_text()
    from ConvAIData import get_label_feature
    get_label_feature()

    from createFeature import LDAFeature
    from Ref_Data import NUM_TOPIC
    LDAFeature(NUM_TOPIC)



if __name__ == "__main__":
    splitTarget('train.csv')
    pipeline()
