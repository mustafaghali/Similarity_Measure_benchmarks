### Similarity measures benchmark
This code compares different similarity measures that are computed on numerical, categorical or mixed features. 
  
   
The selected similarity measures then are compared using linear classifier, the performance metrics are recordered for each individual similarity measure across different datasets.

The classifier data is obtained by constructing similarity functions using similarity measure, for symmetrical functions we take only the upper right triangle of the similarity matrix.  

Depending on the number of classes and samples per each class it's more likely to train the classifer on unblanaced dataset, since the goal is to evaluate the measures and not the classifier we downsampled the data, and then obtained the metrics using k-fold cross validation.
