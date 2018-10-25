import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]
    start_scores = start_scores[:,None].transpose()
    end_scores = end_scores[:,None].transpose()
    best = np.zeros_like(emission_scores)
    backpointer = np.zeros_like(emission_scores, dtype=np.int32)
    best[0] = emission_scores[0] + start_scores[0]
    for i in range(1, emission_scores.shape[0]):
        mmatrix = np.expand_dims(best[i - 1], 1) + trans_scores
        best[i] = emission_scores[i] + np.max(mmatrix, 0)
        backpointer[i] = np.argmax(mmatrix, 0)

    fin= best[-1] + end_scores[0]
    y = [np.argmax(fin)]
    for item in reversed(backpointer[1:]):
        y.append(item[y[-1]])
    y.reverse()
    score = np.max(fin)
    return (score, y)
    