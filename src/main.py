from relevance_metrics import *
from encoding import *
from retrieval import *

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


    # Retrieval

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
    print(tfidf_query(docs,["fat","dog"],get_tfidfs(docs),"ntn"))