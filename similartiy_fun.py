# install py_stringmatching lib
# get_ipython().system(' pip3 install py_stringmatching')


import numpy as np

from scipy.spatial import distance

import py_stringmatching as sm
from py_stringmatching import OverlapCoefficient
from py_stringmatching import Jaccard
from py_stringmatching import MongeElkan
from py_stringmatching import Cosine
from py_stringmatching import Dice
from py_stringmatching import SoftTfIdf
from py_stringmatching import TverskyIndex
from py_stringmatching import Affine


# # Numerical


def cosine (o1,o2):
    """
    cosine similarity function
    input:
    - o1: first object (List)
    - o2: Second object (List)
    output:
    - cosine between 01 and o2
    """
#     return np.arccos(np.dot(o1,o2)/(np.linalg.norm(o1)*np.linalg.norm(o2)))
    o1, o2 = list(o1) , list(o2)
    return Cosine().get_sim_score(o1,o2)



def Euclidean_distance (o1,o2):
    """
    distance similarity function
    input:
    - o1: first object (List)
    - o2: Second object (List)
    output:
    - distance between 01 and o2 float
    """
    o1 , o2= np.array(o1) , np.array(o2)
    return abs(o1-o2).sum()**(0.5)


def wieghted_euclidean (o1,o2,w=0.1):
    """
    wieghted euclidean similarity function
    input:
    - o1: first object (List)
    - o2: Second object (List)
    output:
    - wieghted euclidean distance between 01 and o2 (float)
    """
    o1 , o2= np.array(o1) , np.array(o2)
    return distance.wminkowski(o1,o2,1,w)


def mahalanobis (o1,o2):
    """
    Mahalanobis similarity function
    input:
    - o1: first object (List)
    - o2: Second object (List)
    output:
    - Mahalanobis distance between 01 and o2 (float)
    """
    o1 , o2= np.array(o1) , np.array(o2)
    inv_cov_matrix=np.eye(len(o1))
    return distance.mahalanobis(o1,o2,inv_cov_matrix)

def minkowski (o1,o2):
    """
    minkowski similarity function
    input:
    - o1: first object (List)
    - o2: Second object (List)
    output:
    - minkowski distance between 01 and o2 (float)
    """
    return distance.minkowski(o1,o2)

# # Categorical


def base_line (o1,o2):
    """
    base_line similarity function
    input:
    - o1: first object (List)
    - o2: Second object (List)
    output:
    - 1 if o1=02, 0 otherwise 
    """
    if o1==o2:
        return 1
    else:
        return 0


def overlap (o1,o2):
    """
     overlap similarity function
    input:
    - o1: first object (List)
    - o2: Second object (List)
    output:
    -  measure that measures the overlap between two sets
    """
    o1, o2 = list(o1) , list(o2)
    for i in range (len(o1)):
        o1[i]=str(o1[i])
    for i in range (len(o2)):     
        o2[i]=str(o2[i])
    return OverlapCoefficient().get_sim_score(o1,o2)



def Jaccard_Distance (o1,o2):
    """
     Jaccard similarity function
    input:
    - o1: first object (List)
    - o2: Second object (List)
    output:
    -  Computes Jaccard measure.
    """
    o1, o2 = list(o1) , list(o2)
    for i in range (len(o1)):
        o1[i]=str(o1[i])
    for i in range (len(o2)):     
        o2[i]=str(o2[i])
    return Jaccard().get_sim_score(o1,o2)




def Monge_Elkan(o1,o2):
    """
     Monge Elkan similarity function
    input:
    - o1: first object (List)
    - o2: Second object (List)
    output:
    -  Computes Monge Elkan measure.
    """
    o1, o2 = list(o1) , list(o2)
    for i in range (len(o1)):
        o1[i]=str(o1[i])
    for i in range (len(o2)):     
        o2[i]=str(o2[i])
    return MongeElkan().get_raw_score(o1,o2)



def dice (o1,o2):
    """
     Dice similarity function
    input:
    - o1: first object (List)
    - o2: Second object (List)
    output:
    -  Computes Dice measure.
    """
    o1, o2 = list(o1) , list(o2)
    for i in range (len(o1)):
        o1[i]=str(o1[i])
    for i in range (len(o2)):     
        o2[i]=str(o2[i])
    return Dice().get_sim_score(o1,o2)



def soft_TF_IDF (o1,o2,threshold=0.9):
    """
     soft TF IDF similarity function
    input:
    - o1: first object (List)
    - o2: Second object (List)
    output:
    -  Computes Dice measure.
    """
    o1, o2 = list(o1) , list(o2)
    for i in range (len(o1)):
        o1[i]=str(o1[i])
    for i in range (len(o2)):     
        o2[i]=str(o2[i])
    a=SoftTfIdf(sim_func=Affine().get_raw_score, threshold=threshold)
    return a.get_raw_score(o1,o2)



def tversky_Index (o1,o2):
    """
    tversky_Index  similarity function
    input:
    - o1: first object (List)
    - o2: Second object (List)
    output:
    -  Computes Tversky index similarity.
    """
    o1, o2 = list(o1) , list(o2)
    for i in range (len(o1)):
        o1[i]=str(o1[i])
    for i in range (len(o2)):     
        o2[i]=str(o2[i])
    return TverskyIndex().get_sim_score(o1,o2)

def voting_similarity (o1,o2, numerical=False , categorical=True):
    """
     weight similarity function
    input:
    - o1: first object
    - o2: Second object
    - numerical (pool) True if the data is numerical
    - categorical (pool) True if the data is categorical
    output:
    -  Computes Monge Elkan measure.
    """
    if numerical:
        w=1/4
        score=w * (cosine(o1,o2) + Euclidean_distance(o1,o2) + wieghted_euclidean(o1,o2) + mahalanobis(o1,o2)) 
    else:
        w=1/7
        score=w * (cosine(o1,o2) + overlap(o1,o2) + Jaccard_Distance(o1,o2) + Monge_Elkan(o1,o2) + dice(o1,o2) + soft_TF_IDF(o1,o2) + tversky_Index(o1,o2))
    return score

numerical_similarity_fun= {'cosine':cosine ,  'Euclidean_distance' : Euclidean_distance,
                          'wieghted_euclidean':wieghted_euclidean , 'mahalanobis':mahalanobis , 'minkowski':minkowski}
categorical_similarity_fun = { 'cosine':cosine ,  'overlap':overlap , 'Jaccard_Distance':Jaccard_Distance , 'Monge_Elkan':Monge_Elkan ,
                              'dice':dice , 'soft_TF_IDF':soft_TF_IDF , 'tversky_Index':tversky_Index }
