import codecs
import json
from os import listdir, remove
from os.path import isfile, join

def load_d2v_base():
    file = codecs.open('d2v_base.bin', 'r', 'utf-8')
    vectors = json.loads(file.read())
    file.close()
    return vectors
   
def load_w2v_base():
    file = codecs.open('w2v_base.bin', 'r', 'utf-8')
    vectors = json.loads(file.read())
    file.close()
    return vectors

def load_indexing(force=False):
    path = './database/'
    index_file = 'indexing.txt'

    if not isfile(index_file) or force:
        files = [f for f in listdir(path) if isfile(join(path, f))]
        f = open(index_file, 'w')
        for filename in files:
            jf = join(path, filename)
            file = codecs.open(jf, 'r', 'utf-8')
            file = json.load(file)
            if 'text' in file:
                f.write(filename + '\n')
            else:
                remove(jf)
        f.close()
    
    files = [f.strip() for f in open(index_file, 'r').readlines()]
    return files


def paragraphs(docs):
    for index, doc in enumerate(docs):
        for par in doc['paragraphs']:
            yield index, par
