from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words
import nltk,os,string,copy,pickle,re,math
import numpy as np
import pandas as pd
import warnings
def stemming(data):
    stemmer = PorterStemmer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text
def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text
def convert_lower_case(data):
    return np.char.lower(data)
def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = stemming(data)
    return data


with open('CISI.ALL') as f:
    lines = ""
    for l in f.readlines():
        lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
    lines = lines.lstrip("\n").split("\n")

doc_set = {}
doc_id = ""
doc_text = ""
for l in lines:
    if l.startswith(".I"):
        doc_id = l.split(" ")[1].strip()
    elif l.startswith(".X"):
        doc_set[doc_id] = doc_text.lstrip(" ")
        doc_id = ""
        doc_text = ""
    else:
        doc_text += l.strip()[3:] + " " # The first 3 characters of a line can be ignored.
        doc_text=preprocess(doc_text)

print(f"Number of documents = {len(doc_set)}" + ".\n") # printing the total no of documents.)
N=len(doc_set)

processed_text = []
processed_title = []

for i in doc_set.values():
    text = i
    processed_text.append(word_tokenize(str(preprocess(text))))
    processed_title.append(word_tokenize(str(preprocess(text))))

DF = {}
for i in range(N):
    tokens = processed_text[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}

    tokens = processed_title[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}
for i in DF:
    DF[i] = len(DF[i])
total_vocab= len(DF)

def doc_freq(word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c

# TF-IDF
doc = 0
tf_idf = {}
for i in range(N):
    tokens = processed_text[i]
    counter = Counter(tokens + processed_title[i])
    words_count = len(tokens + processed_title[i])
    for token in np.unique(tokens):
        tf = counter[token] / words_count
        df = doc_freq(token)
        idf = np.log((N + 1) / (df + 1))
        tf_idf[doc, token] = tf * idf
    doc += 1

with open('CISI.QRY') as g:
    qlines = ""
    for m in g.readlines():
        qlines += "\n" + m.strip() if m.startswith(".") else " " + m.strip()
    qlines = qlines.lstrip("\n").split("\n")

qry_set = {}
qry_id = ""
qry_text=""
for m in qlines:
    if m.startswith(".I"):
        qry_id = m.split(" ")[1].strip()
    elif m.startswith(".W"):
        qry_set[qry_id] = m.strip()[3:]
        qry_id = ""
squery=qry_set["1"]

def matching_score(k, query):
    preprocessed_query = preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))
    print("\nQuery:", query)
    print("Matching Score")
    print("")
    #print(tokens)
    query_weights = {}
    for key in tf_idf:
        if key[1] in tokens:
            try:
                query_weights[key[0]] += tf_idf[key]
            except:
                query_weights[key[0]] = tf_idf[key]
    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)
    print("")
    l = []
    for i in query_weights[:10]:
        l.append(i[0])
    print("Index ",l)
matching_score(10,squery)

def cosine_sin(a, b):        #TF-IDF cosine similarity
    try:
        cos_sin = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
        return cos_sin
    except:
        np.seterr()
warnings.filterwarnings("ignore", category=RuntimeWarning)
D = np.zeros((N, total_vocab))
for i in tf_idf:
    try:
        ind = total_vocab.index(i[1])
        D[i[0]][ind] = tf_idf[i]
    except:
        pass

def gen_vector(tokens):
    Q = np.zeros(total_vocab)
    counter = Counter(tokens)
    words_count = len(tokens)
    query_weights = {}
    for token in np.unique(tokens):

        tf = counter[token] / words_count
        df = doc_freq(token)
        idf = math.log((N + 1) / (df + 1))
        try:
            ind = total_vocab.index(token)
            Q[ind] = tf * idf
        except:
            pass
    return Q
def cosine_similarity(k, query):      # function which calculate the cosine similarity
    print("Cosine Similarity")
    preprocessed_query = preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))
    print("\nQuery:", query)
    print("")
    print(tokens)
    d_cosines = []
    query_vector = gen_vector(tokens)
    for d in D:
        d_cosines.append(cosine_sin(query_vector, d))
    out = np.array(d_cosines).argsort()[-k:][::-1]
    print("")
    print(out)
    df = pd.DataFrame(doc_set.items())
    for i in range(10):
        j = out[i]
        print("  ",df.iloc[j-1,0],"--",df.iloc[j-1,1])
cosine_similarity(10,squery)
