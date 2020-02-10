### Similarity measures benchmark
This code compares different similarity measures computed on numerical, categorical or mixed features. 
  
   
The selected similarity measures then compared using non linear classifier, the performance metrics are recordered for each individual similarity measure across different datasets.

The classifier data is obtained by constructing similarity matrices with kernels of these similarity measure, to avoid redundancy we take only the upper right triangle of symmetric matrices. 

Depending on the number of classes and samples per each class it's more likely to train the classifer on unblanaced dataset, since the goal is to evaluate the similarity measures and not the classifier itself we downsampled the data, and then we use k-fold cross validation to validate the classifer performance.
