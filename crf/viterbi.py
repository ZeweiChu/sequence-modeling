import numpy as np


def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    y = []
    scores = 0.0 + start_scores.reshape((L, 1)) # L x 1 
    prevs = []
    for i in range(N-1):
        emit_scores = emission_scores[i].reshape((L, 1)) # L x 1
        scores = scores + emit_scores + trans_scores # L x L

        prevs.append(scores.argmax(0))
        scores = scores.max(0).reshape((L, 1)) # L x 1 
    
    scores = (scores.reshape((L, 1)) + end_scores.reshape((L, 1)) + emission_scores[N-1].reshape((L, 1))).reshape((L)) # L 
    
    y = [scores.argmax()]
    score = scores.max()
       
    for indices in prevs[::-1]:
        last_index = y[0]
        new_index = indices[last_index]
        y = [new_index] + y
    
    return [score, y]
