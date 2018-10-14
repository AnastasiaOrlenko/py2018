from flask import Flask, request, redirect
from load_functions import *
from gensim import matutils
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import numpy as np
import re
import pymorphy2
import urllib.parse

w2v_path = 'models/ru.bin'
d2v_path = 'models/d2v.bin'
bm25_path = 'models/bm25.bin'
app = Flask(__name__)

from bm25.query import QueryProcessor
from bm25.parse import Corpus

def search_inv_index(document):
    results = models['bm25'].run_query(get_words(document))
    results = [(int(key), val / 10) for key, val in results.items()]
    return results

def similarity(v1, v2):
    v1_norm = matutils.unitvec(np.array(v1))
    v2_norm = matutils.unitvec(np.array(v2))
    return np.dot(v1_norm, v2_norm)

w = re.compile('[A-zА-я]+')
morph = pymorphy2.MorphAnalyzer()

def get_words(text):
    words = w.findall(text)
    words = [morph.parse(w.lower())[0].normal_form for w in words]
    return words

def count_words(text):
    words = {}
    indoc = get_words(text)
    for word in indoc:
        word = word.lower()
        if word not in words:
            words[word] = 0
        words[word] = words[word] + 1
    return words
   
def get_w2v_vector(words):
    vec = [0 for i in range(models['w2v'].vector_size)]
    size = 0
    for word, count in words.items():
        if word not in models['w2v'].wv:
            continue
        size += count
        wv = models['w2v'].wv[word]
        vec = [vec[i] + wv[i] * count for i in range(len(vec))]
    if size == 0:
        return None
    return [i / size for i in vec]

def search_w2v(document):    
    vec = get_w2v_vector(count_words(document))
    if not vec:
        return []
    results = [(w2v_vec[0], similarity(w2v_vec[1], vec)) for w2v_vec in vectors['w2v']]
    return results

def search_d2v(document):
    vec = models['d2v'].infer_vector(get_words(document), epochs=10)
    vec = [float(f) for f in vec]
    results = [(d2v_vec[0], similarity(d2v_vec[1], vec)) for d2v_vec in vectors['d2v']]
    return results


vectors = {}
docs = []
models = {}
files = []

@app.route("/reset")
def reset():
    vectors['d2v'] = load_d2v_base()
    vectors['w2v'] = load_w2v_base()
    files.clear()
    files.extend(load_indexing())
    #models['w2v'] = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    models['w2v'] = Word2Vec.load(w2v_path)
    models['d2v'] = Doc2Vec.load(d2v_path)
    models['bm25'] = QueryProcessor.load(bm25_path)

@app.route("/")
def hello ():
    page  = '<form action="/query" method="post" enctype="application/json">'
    page += '<input type="text" name="search" required>'
    page += '<select name="search_method">'
    page += '<option value="inverted_index">Inverted index (BM25)</option>'
    page += '<option value="word2vec" selected>Word2Vec</option>'
    page += '<option value="doc2vec">Doc2vec</option>'
    page += '</select>'
    page += '<input type="submit" value="Search">'
    page += '</form>'
    return page

@app.route("/query", methods=["POST"])
def query():
    data = request.get_data()
    data = urllib.parse.unquote(data.decode('utf-8'))
    form = {}
    for line in data.split('&'):
        line = line.split('=')
        form[line[0]] = line[1]
    next = form['search_method'] + '?text=' + form['search']
    return redirect('/query/' + next, code=302)

@app.route("/base/<index>")
def getfile(index):
    try:
        file = files[int(index)]
        file = codecs.open(join('./database/', file), 'r', 'utf-8')
        doc = json.load(file)
        page  = '<h1>' + doc['title'] + '</h1>'
        page += '<h2>Цена: ' + str(doc['price']) + '</h2><br>'
        page += '<p>' + doc['text'].replace('\n', '<br>') + '</p>'
        return page
    except Exception as e:
        
        return str(e) + "<br>No such document: " + index

last_search = { 'text': None, 'method': None , 'result': None }

@app.route("/query/<search_method>")
def search(search_method):
    text = request.args.get('text')
    try:
        pad = int(request.args.get('pad'))
        if pad < 0: pad = 0
    except:
        pad = 0
    
    if last_search['text'] != text or last_search['method'] != search_method:
        if search_method == 'inverted_index':
            search_result = search_inv_index(text)
        elif search_method == 'word2vec':
            search_result = search_w2v(text)
        elif search_method == 'doc2vec':
            search_result = search_d2v(text)
        else:
            raise TypeError('unsupported search method')
        search_result.sort(key = lambda x: x[1], reverse=True)
        last_search['text'] = text
        last_search['method'] = search_method
        last_search['result'] = search_result
    else:
        search_result = last_search['result']
    
    if len(search_result) == 0:
        return 'Нет совпадений'
    page = ''
    for i in range(10 * pad, min(10 * (pad+1), len(search_result))):
        file = files[search_result[i][0]]
        file = codecs.open(join('./database/', file), 'r', 'utf-8')
        doc = json.load(file)
        page += '<a href="../../base/' + str(search_result[i][0]) + '">'
        page += doc['title'] + '</a><br>'
        page += '<p>Совпадение: ' + str(search_result[i][1]*100) + '%</p><br><br><br>'
    
    if pad > 0:
        page += '<a href="' + request.base_url + '?pad=' + str(pad-1) + '&text=' + text + '">Prev</a>    '
    if 10 * (pad+1) < len(search_result) - 1:
        page += '<a href="' + request.base_url + '?pad=' + str(pad+1) + '&text=' + text + '">Next</a>'

    return page
    
reset()
