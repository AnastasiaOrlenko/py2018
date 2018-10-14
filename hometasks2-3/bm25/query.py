__author__ = 'Nick Hirakawa'

try:
    from invdx import *
    from rank import score_BM25
except:
    from bm25.invdx import *
    from bm25.rank import score_BM25

import operator
import codecs

class QueryProcessor:
    def __init__(self, corpus=None):
        if corpus:
            self.index, self.dlt = build_data_structures(corpus.corpus)

    def run(self, queries):
        results = []
        for query in queries:
            results.append(self.run_query(query))
        return results

    def run_query(self, query):
        query_result = dict()
        for term in query:
            if term in self.index:
                doc_dict = self.index[term] # retrieve index entry
                for docid, freq in doc_dict.items(): #for each document and its word frequency
                    score = score_BM25(n=len(doc_dict), f=freq, qf=1, r=0, N=len(self.dlt),
                                       dl=self.dlt.get_length(docid), avdl=self.dlt.get_average_length()) # calculate score
                    if docid in query_result: #this document has already been scored once
                        query_result[docid] += score
                    else:
                        query_result[docid] = score
        return query_result

    def save(self, path):
        file = codecs.open(path, 'w', 'utf-8')
        file.write(str(self.index.index))
        file.write('\n')
        file.write(str(self.dlt.table))
        file.flush()
        file.close()
    
    def load(path):
        import ast
        self = QueryProcessor()
        file = codecs.open(path, 'r', 'utf-8')
        
        self.index = InvertedIndex()
        self.index.index = ast.literal_eval(file.readline())
        
        self.dlt = DocumentLengthTable()
        self.dlt.table = ast.literal_eval(file.readline())
        file.close()
        return self
