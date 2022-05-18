from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.stem.isri import ISRIStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import arabic_reshaper
import nltk as n
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import _pickle as cPickle
import joblib



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
def readdata(url):
    data = pd.read_excel(url)
    return data
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

def preprocessinput(X):
    st = ISRIStemmer()
    stemmed_sentences = []
    for sentence in X:
        list_of_words = []

        for a in word_tokenize(sentence):
            list_of_words.append(st.stem(a))

        stemmed_sentences.append(' '.join(list_of_words))

    return stemmed_sentences
def vectorizer(X):

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X).toarray()

    return X,vectorizer

def splitdata(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

data=readdata("query_result_2022-05-16T08_26_15.499808Z.xlsx")
sanaasdata=readdata("Book1.xlsx")
stopwords_arabic=readstopdatafile()
X1=sanaasdata['Title']
X=data['stories_headlines']
X2=data['stories_content']
X=X+" "+X2
X=X.append(X1, ignore_index=True)
y=data['country']
y1=sanaasdata['country']
y=y.append(y1, ignore_index=True)

y,targe_code=preprocessoutput(y)
X=X.apply(lambda s: preprocessText(s,stopwords_arabic))

X=preprocessinput(X)
X,vectorizer=vectorizer(X)
X_train, X_test, y_train, y_test = splitdata(X,y)
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))



print(accuracy_score(y_test, y_pred))
joblib.dump(classifier,'model2.joblib')
with open('vectorizer3.pk', 'wb') as fin:
    cPickle.dump(vectorizer, fin)

