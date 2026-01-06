import numpy as np
from scipy.stats import spearmanr

def spearman_stability(exp1, exp2):
    corr, _ = spearmanr(exp1, exp2)
    return corr

def top_k_jaccard(exp1, exp2, k=5):
    idx1 = set(np.argsort(exp1)[-k:])
    idx2 = set(np.argsort(exp2)[-k:])
    return len(idx1 & idx2) / len(idx1 | idx2)
