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
