import numpy as np


# A = [  0.0038203,     0.98265,    0.013529]
# B = [   0.001952,   0.0032686,     0.99478]
# C = [    0.01646,     0.13862,     0.84492]
# D = [  0.0019251,     0.99385,   0.0042294] 
# E=  [ 2.1224e-05,     0.99706,    0.002916] 
# F = [  0.0027917,      0.9849,    0.012305] 
# G = [  0.0045303,   0.0046658,      0.9908] 
# H = [    0.90336,    0.094251,   0.0023868] 
# I = [     0.6317,     0.35902,   0.0092771] 
# J = [  0.0057923,     0.99228,   0.0019275] 
# K = [  0.0024267,     0.99336,   0.0042085] 
# L = [   0.012543,     0.97233,    0.015124]

A = [  0.0019413,   0.0029807,     0.99508] 
B = [  0.0018634,   0.0021047,     0.99603] 
C =[  0.0023809,   0.0048565,     0.99276] 
D =[  0.0027724,   0.0030224,     0.99421]

cell_probs = np.array([A, B, C, D])

def class_weighted_pooling(cell_probs, class_weights=[3.0, 1.5, 1.0], pooling='mean'):
    if len(cell_probs) == 0:
        return None
    
    weighted_probs = cell_probs * np.array(class_weights)
    print("Weighted probs:\n", weighted_probs)
    
    if pooling == 'max':
        slide_probs_weighted = np.max(weighted_probs, axis=0)
    elif pooling == 'mean':
        slide_probs_weighted = np.mean(weighted_probs, axis=0)
    else:
        raise ValueError(f"Unknown pooling: {pooling}")
    
    print("Slide probs weighted (before normalization):\n", slide_probs_weighted)
    slide_probs = slide_probs_weighted / slide_probs_weighted.sum()
    print("Slide probs weighted (after normalization):\n", slide_probs)
    return slide_probs

class_weighted_pooling(cell_probs, class_weights=[3.0, 1.5, 1.0], pooling='mean')
print("\n---\n")
class_weighted_pooling(cell_probs, class_weights=[3.0, 1.5, 1.0], pooling='max')