##### RELEVANCE


from math import log2
from typing import List, Tuple, Union



""" the relevance of a document """
Relevance = int

""" a ranked list of entries (by position in the list) """
Query = List[Relevance]


#### BINARY RELEVANCE 




def true_positives(query : Query):
    """ counts the number of documents in the query which are relevant (non-zero relevance) """
    return sum([1 for rel in query if rel > 0])

def false_positives(query: Query):
    """ countes the number of documents in the query which are not relevant (zero relevance)"""
    return sum([1 for rel in query if rel <= 0])



def precision_at_k(query : Query, k : int):
    """ calculates the precision after cutting off query at k results """

    assert isinstance(query,list)
    assert len(query) >= k 
    assert len(query) > 0

    trimmed_query = query[:k]
    tp = true_positives(trimmed_query)
    fp = false_positives(trimmed_query)

    return tp / (tp + fp)

def r_precision(query: Query, r : int = None):
    """ calculates the r precision for the given query, assuming r is the amount of relevant documents. If 
        r is None, all relevant documents are assumed to be in the retrieved query """
    if r:
        return precision_at_k(query, r)
    else:
        return precision_at_k(query, true_positives(query)) # if all the relevant 

    
def average_precision(query : Query, r : int = None):
    assert len(query) > 0

    """ calculates the average precision for the given query, where r is the number of known relevant documents. 
        if r is None, it is calculated assuming all relevant docs are in the query """
    if not r:
        r = true_positives(query)

    ap = 0
    for i in range(len(query)):
        ap += precision_at_k(query,i+1) * int(bool(query[i]))

    return ap / r




def mean_average_precision(queries : List[Query],rs = List[int]):
    # we need r values for each query
    assert len(queries) == len(rs)
    assert len(queries) > 0

    map = 0
    for q,r in zip(queries,rs):
        map += average_precision(q,r)
    return map / len(queries)

#### GRADED RELEVANCE

def discounted_cumulative_gain_at_k(query: Query, k : int):
    assert len(query) >= k
    assert len(query) > 0
    dcg = query[0]
    for i,r in enumerate(query[1:],start = 1):
        dg = r / log2(i+1)
        dcg += dg

    return dcg

def normalised_discounted_cumulative_gain_at_k(query : Query, k : int):
    ideal_query = sorted(query,reverse=True)
    return discounted_cumulative_gain_at_k(query,k) / discounted_cumulative_gain_at_k(ideal_query,k)