# install py_stringmatching lib
# get_ipython().system(' pip3 install py_stringmatching')


import numpy as np
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
    return TverskyIndex().get_sim_score(o1,o2)

numerical_similarity_fun= {'cosine':cosine ,  'Euclidean_distance' : Euclidean_distance}
categorical_similarity_fun = { 'cosine':cosine ,  'overlap':overlap , 'Jaccard_Distance':Jaccard_Distance , 'Monge_Elkan':Monge_Elkan ,
                              'dice':dice , 'soft_TF_IDF':soft_TF_IDF , 'tversky_Index':tversky_Index }

