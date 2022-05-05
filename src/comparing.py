

from math import log2
from typing import Callable, List
from retrieval import *



""" the class of a document """
DocClass = int

""" Binary class-term metrix """
ClassTermMatrix = Dict[str,int]

""" Document class assignment """
DocClasses = Dict[DocID, DocClass]


""" Comparison focus, for binary matrices """
ComparisonFocus = Tuple[str,DocClass] # positive,positive


""" type of metric for class-term rankings """
ClassTermRanking =  Dict[DocClass,Dict[str,float]]

def get_binary_class_term_matrix(
    docs: List[Document], doc_classes : DocClasses, 
    comparison_focus : ComparisonFocus
    ):
    """ builds a binary class-term matrix for further computation, inefficient but who cares,
         samples are gonna be small anyway """
         
    POSITIVE_TERM = comparison_focus[0]
    POSITIVE_CLASS = comparison_focus[1]

    class_term_matrix = defaultdict(int)

    for id,words in docs:
        
        doc_class = doc_classes[id]

        if doc_class == POSITIVE_CLASS:
            if POSITIVE_TERM in words:
                class_term_matrix["N11"] += 1
            else:
                class_term_matrix["N01"] += 1
        else:
            if POSITIVE_TERM in words:
                class_term_matrix["N10"] += 1
            else:
                class_term_matrix["N00"] += 1

    class_term_matrix["N"] = len(docs)

    return class_term_matrix

def mutual_information(bctm : ClassTermMatrix):
    """ computes the mutual information for a class-term pair given it's count matrix"""
    
    def term(left_sum, bottom_sum):
        left_side = left_sum / bctm["N"] 

        if left_side == 0:
            return 0 
        else:
            return (left_sum/bctm["N"]) * log2((bctm["N"] * left_sum)/bottom_sum)

    total = 0
    total += term(bctm["N11"],(bctm["N10"]+bctm["N11"]) * (bctm["N10"]+bctm["N11"]))
    total += term(bctm["N01"],(bctm["N01"]+bctm["N00"]) * (bctm["N01"]+bctm["N11"]))
    total += term(bctm["N10"],(bctm["N10"]+bctm["N11"]) * (bctm["N00"]+bctm["N10"]))
    total += term(bctm["N00"],(bctm["N00"]+bctm["N01"]) * (bctm["N00"]+bctm["N10"]))

    return total

def chi_squared(bctm : ClassTermMatrix):

    total = 0
    total += (bctm["N11"] + bctm["N10"] + bctm["N01"] + bctm["N00"]) * (((bctm["N11"] * bctm["N00"]) - (bctm["N10"] * bctm["N01"]))**2)
    total /= (bctm["N11"] + bctm["N01"]) * (bctm["N11"] + bctm["N10"]) * (bctm["N10"] + bctm["N00"]) * (bctm["N01"] + bctm["N00"])
    return total

def rank_terms_by_stat(documents : List[Document], 
    doc_classes: DocClasses,
    stat: Callable) -> ClassTermMatrix:

    vocab = get_vocabulary(documents)
    classes = set(doc_classes.values())
    out = defaultdict(dict)

    for t in vocab:
        for c in classes:
            matrix = get_binary_class_term_matrix(documents,doc_classes,(t,c))
            out[c][t] = stat(matrix)

    return out



def print_class_term_ranking(class_term_ranking : ClassTermRanking):
    print()
    for c in class_term_ranking.keys():
        print(f"\n----- class term ranking for class {c} ----- ")
        for k,v in sorted(class_term_ranking[c].items(),key=lambda a: a[1],reverse=True):
            print(f"Term: {k}, Score:{v}")
