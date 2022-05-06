import numpy as np
from relevance_metrics import *
from encoding import *
from retrieval import *
from comparing import *
from classification import *
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix 
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

if __name__ == "__main__":
    ### Playground

    ## Q1

    # a)
    # b)
    # c)
    # d)
    # e)
    # f)

    ## Q2

    # a)
    # b)
    # c)
    # d)
    # e)
    # f)

    ## Q3

    # a)
    # b)
    # c)
    # d)
    # e)
    # f)


    ## Examples from the slides

    # precision at k
    print(precision_at_k([0,1,0,1,0,1,0,1,0,1],5))

    # r precision
    print(r_precision([0,1,0,1,0,1,0,1,0,1]))

    # AP slides
    ap_q_1 = [1,1,0,0,1,0,0,0,1,0]
    ap_q_2 = [0,0,1,0,0,0,1,0]
    ap_q_3 = [0,1,0,0,1,0,0,1]
    print(average_precision(ap_q_1))
    print(average_precision(ap_q_2, r = 3))
    print(average_precision(ap_q_3, r = 7))

    # MAP slides
    print(mean_average_precision([ap_q_1,ap_q_2,ap_q_3],[None,3,7]))

    # DCG slides
    dcg_q = [3,2,3,0,0,1,2,2,3,0]
    print(discounted_cumulative_gain_at_k(dcg_q,10))
    print(normalised_discounted_cumulative_gain_at_k(dcg_q,10))

    print(average_precision([0,1,0,1,0,1],6))
    print(normalised_discounted_cumulative_gain_at_k([0,1,0,1,0,1],6))

    # encoding slides

    # slide is wrong on the delta encoding of 100011 - 100019
    print(delta_encode([100002,100007,100008,100011,100019]))
    print(delta_decode([100002,5,1,3,8]))

    vbyte_vals = [5,130,7]
    print_in_binary(vbyte_encode(vbyte_vals))
    print(vbyte_decode(vbyte_encode(vbyte_vals)))

    # example - read vbyte + delta encoded list, then decode it
    bits = bitstring_to_bytes("100001100000001110000101000000011000000110000110")
    print_in_binary(bits)
    print(delta_decode(vbyte_decode(bits)))


    ## Retrieval

    # document format
    d1 = (1,["he","likes","to","wink","and","drink"])
    d2 = (2,["he","likes","to","drink"])

    print(get_vocabulary([d1,d2]))
    print(get_tfidfs([d1,d2]))

    # jacard coefficient slides

    print(jaccard_coefficient(d1,d2))


    # example of tfidf query (q2 in 2020 may)

    d1 = (1,["fat","dog","fat","dog"])
    d2 = (2,["run","watch","dog","job"])
    d3 = (3,["fat","fat","fat","dog"])
    d4 = (4,["fat","dog","fat","go"])
    d5 = (5,["job","fat","fat","dog"])
    d6 = (6,["go","fat","fat","job"])
    d7 = (7,["watch","job","run","dog"])
    d8 = (8,["nice","nice","lovely","dog"])
    docs = [d1,d2,d3,d4,d5,d6,d7,d8]
    print(get_tfidfs(docs))
    print(tfidf_query(docs,["fat","dog"],get_tfidfs(docs),"ntn"))


    ## comparison

    # 2021 q7
    d1 = (1,["hop","frog","nice","frog"])
    d2 = (2,["frog","pond","watch","hop"])
    d3 = (3,["good","good","good","frog"])
    d4 = (4,["good","nice","frog","go"])
    d5 = (5,["pond","good","good","frog"])
    d6 = (6,["go","good","good","pond"])
    d7 = (7,["watch","pond","hop","frog"])
    d8 = (8,["nice","nice","quiet","pond"])
    docs = [d1,d2,d3,d4,d5,d6,d7,d8]

    doc_classes = {1:0,2:0,3:0,4:0,5:1,6:1,7:1,8:1}
    # mutual information

    print_class_term_ranking(rank_terms_by_stat(docs,doc_classes,mutual_information))
    print_class_term_ranking(rank_terms_by_stat(docs,doc_classes,chi_squared))


    ## Classification 

    # metrics q6 2021

    conf_matrix = [[500,1,9,2],
                    [37,86,5,100],
                    [62,14,95,2],
                    [49,50,6,120]]
    
    conf_matrix = confusion_matrix(conf_matrix)

    class_labels = [0,1,2,3]

    # wee hack to reduce amount of code
    y_true,y_pred = get_fake_sklearn_true_and_pred(conf_matrix)

    print(conf_matrix)
    print(sklearn_confusion_matrix(y_true,y_pred))

    # we can now do all kinds of shit with sklearns metrics
    print(accuracy_score(y_true,y_pred))

    ## micro-averaged scores
    # identical to accuracy for micro
    print(f1_score(y_true,y_pred,labels=class_labels,average="micro"))
    print(recall_score(y_true,y_pred,labels=class_labels,average="micro"))
    print(precision_score(y_true,y_pred,labels=class_labels,average="micro"))

    ## macro-averaged scores
    # different in macro
    print(f1_score(y_true,y_pred,labels=class_labels,average="macro"))
    print(recall_score(y_true,y_pred,labels=class_labels,average="macro"))
    print(precision_score(y_true,y_pred,labels=class_labels,average="macro"))

    
    ## Query correction

    # rochios algorithm 2020 q 2

    query = [0.2,0.06,0,0,0,0,0,0]
    rel = [0,0.06,0.6,0.6,0.3,0,0,0]
    irel = [0.4,0.12,0,0,0,0,0,0]

    print(rocchios_algorithm(query,[rel],[irel],1,1,1))