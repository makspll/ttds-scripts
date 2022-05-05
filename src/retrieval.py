from collections import defaultdict
from dataclasses import dataclass, field
from math import log10, sqrt
from typing import Dict, List, Set, Tuple




""" the type of document identifier """
DocID = int

""" the type of a document """
Document = Tuple[DocID,List[str]]

""" vocabulary type """
Vocabulary = Set[str]

""" tfidf bank type """
Tfidf = Dict[str,"TFIDFEntry"]

""" Query results type """
QueryResults = Dict[DocID, float]

@dataclass(eq=True,repr=False)
class TFIDFEntry():
    token : str = field(repr=False)
    N : int = field(repr=False,compare=False)
    df : int = field(default=0,compare=False)
    tf : Dict[DocID,int] = field(default_factory = lambda: defaultdict(int),compare=False)

    def __hash__(self) -> int:
        return hash(self.token)

    def __eq__(self, __o: object) -> bool:
        return self.token.__eq__(__o)

    def __repr__(self):
        return f"\tdf: {self.df}, idf: {self.idf:.3f}, \ttf: {','.join([str(k)+':'+str(v) for k,v in self.tf.items()])}\n"

    @property
    def idf(self):
        return log10(self.N / self.df) 

def get_vocabulary(documents : List[Document]):

    vocab = set()
    for (id,words) in documents:
        for w in words:
            vocab.add(w)

    return vocab


def get_tfidfs(documents: List[Document]) -> Tfidf:
    tfidf = {}

    for id,words in documents:
        seen_words = set()
        for w in words:
            
            entry = tfidf.setdefault(w,TFIDFEntry(w,len(documents)))
            if w not in seen_words:
                entry.df += 1
            entry.tf[id] += 1
            seen_words.add(w)

    return tfidf

def jaccard_coefficient(doc1 : Document, doc2: Document):
    set1 = set(doc1[1])
    set2 = set(doc2[1])

    return len(set1.intersection(set2)) / len(set1.union(set2))

def tfidf_query(documents : List[Document], query: List[str], tfidfs: Tfidf, smart_variant="ltn") -> QueryResults:
    """ performs a tfidf query, not exactly aligned with the SMART retrieval system, but should work for exam questions, 
        particularly normalisation is different, there is no notion of vectorisation"""
    query_scores = defaultdict(float)

    query_vocab = set(query)

    for id,words in documents:
        document_vocab = set(words)
        query_scores[id] = 0

        weights = []
        for w in document_vocab.intersection(query_vocab):

            if smart_variant[0] == "n":
                tf = tfidfs[w].tf[id]
            elif smart_variant[0] == "l":
                tf = (1 + log10(tfidfs[w].tf[id])) 
            elif smart_variant[0] == "b":
                tf = 1 if tfidfs[w].tf[id] > 0 else 0
            else:
                raise Exception("Smart variant not supported")
            
            if smart_variant[1] == "n":
                idf = 1
            elif smart_variant[1] == "t":
                idf = tfidfs[w].idf
            else: 
                raise Exception("Smart variant not supported")


            weights.append(tf * idf) 

        if smart_variant[2] == "n":
            query_scores[id] = sum(weights)
        elif smart_variant[2] == "c":
            query_scores[id] = 1/sqrt(sum([x**2 for x in weights]))
        else:
            raise Exception("Smart variant not supported")

    return query_scores
