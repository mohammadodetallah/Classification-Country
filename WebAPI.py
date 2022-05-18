
import flask as fs
import numpy as np
import tensorflow as tf
from nltk.stem.isri import ISRIStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import arabic_reshaper
import nltk as n
import joblib
import _pickle as cPickle
def removeStopWords(text,stopwords):
    text_tokens = word_tokenize(text)
    return " ".join([word for word in text_tokens if not word in stopwords])

def removePunctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return " ".join(tokenizer.tokenize(text))

def preprocessText(text,stopwords,wordcloud=False):

    noStop=removeStopWords(text,stopwords)

    noPunctuation=removePunctuation(noStop)

    if wordcloud:
        text=arabic_reshaper.reshape(noPunctuation)
        #text=get_display(text)
        return text
    return noPunctuation
def readstopdatafile():
    file1 = open('stopwordsarabic.txt', 'r', encoding='utf-8')
    stopwords_arabic = file1.read().splitlines()
    return stopwords_arabic
def preprocessoutput(y):
    target_names = y.unique()
    target_codes = {n: i for i, n in
                    enumerate(target_names)}

    y = y.map(target_codes)
    return y,target_codes


def prede(model,title,countv):

    X = preprocessText(title, stopwords_arabic)
    token = n.word_tokenize(X)
    st = ISRIStemmer()
    list_of_words = []
    for a in token:

        list_of_words.append(st.stem(a))

    X=[' '.join(list_of_words)]
    X=countv.transform(X)

    y_pred = model.predict(X)
    switcher ={0:'kuwait', 1:'saudi arabia', 2:'egypt', 3:'jordan', 4:'emirates', 5:'Qatar', 6:'bahrain', 7:'iraq', 8:'palestine', 9:'lebanon', 10:'English'}
    y_pred=switcher.get(y_pred[0])
    return y_pred


app = fs.Flask(__name__)
stopwords_arabic = readstopdatafile()
@app.route("/predict",methods=['POST'])
def predict():

    model = joblib.load("model2.joblib")
    with open("vectorizer3.pk", "rb") as input_file:
        v = cPickle.load(input_file)

    graph = tf.compat.v1.get_default_graph()
    parameters = []
    parameters.append(fs.request.json["Headlines"])
    inputs = np.asarray(parameters)
    print(inputs[0])
    with graph.as_default():
        pred = prede(model,inputs[0],v)
        print(pred)
    return fs.jsonify({"Country":str(pred)})

if __name__ == "__main__" :
    app.run(port=500, debug=True)

