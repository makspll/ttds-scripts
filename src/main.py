from relevance_metrics import *


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