import math
import os
from google.colab import drive
drive.mount('/content/drive')

corpus_path = "/content/drive/MyDrive/Corpus"  # update the path to your Corpus directory
documents = []
def parse_documents(documents):
    index = {}
    
    doc_lengths = {}
    for i, doc in enumerate(documents):
        terms = doc.lower().split()
        doc_weights = {}
        for term in terms:
            if term not in index:
                index[term] = []
            if len(index[term]) > 0 and index[term][-1][0] == i:
                index[term][-1] = (i, index[term][-1][1] + 1)
            else:
                index[term].append((i, 1))
            if term not in doc_weights:
                doc_weights[term] = 0
            doc_weights[term] += 1
        doc_weights = normalize_doc_weights(doc_weights, len(terms))
        for term, weight in doc_weights.items():
            index[term][-1] = (i, weight)
        doc_lengths[i] = sum([weight ** 2 for weight in doc_weights.values()])
    
    return index, doc_lengths

def compute_idf(index, num_documents):
    idf = {}
    for term in index:
        idf[term] = math.log10(num_documents / len(index[term]))
    return idf

def compute_query_weights(query, idf):
    query_weights = {}
    terms = query.lower().split()
    for term in terms:
        if term not in query_weights:
            query_weights[term] = 0
        query_weights[term] += 1
    for term in query_weights:
        if term in idf:
            query_weights[term] *= idf[term]
    return query_weights

def normalize_doc_weights(doc_weights, doc_length):
    for term in doc_weights:
        doc_weights[term] /= doc_length
        #print(doc_weights)
    return doc_weights
def compute_cosine_similarity(query_weights, doc_weights):
    dot_product = 0
    for term in query_weights:
        if term in doc_weights:
            dot_product += query_weights[term] * doc_weights[term]
    query_norm = math.sqrt(sum([weight ** 2 for weight in query_weights.values()]))
    doc_norm = math.sqrt(sum([weight ** 2 for weight in doc_weights.values()]))
    if query_norm == 0 or doc_norm == 0:
        return 0
    return dot_product / (query_norm * doc_norm)

def rank_documents(query_weights, index, doc_lengths):
    doc_scores = {}
    for term in query_weights:
        if term not in index:
            continue
        for posting in index[term]:
            doc_id, tf = posting
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0
            doc_scores[doc_id] += query_weights[term]*(1 + math.log10(tf)) * math.log10(len(doc_lengths) / len(index[term]))
            
    for doc_id in doc_scores:
        doc_scores[doc_id] /= doc_lengths[doc_id]
    
    ranked_docs = sorted(
        doc_scores.items(),
        key=lambda x: (-x[1], x[0])
    )
    return [doc_id for doc_id, score in ranked_docs]
