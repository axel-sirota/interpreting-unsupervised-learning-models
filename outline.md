# Interpreting Unsupervised Learning Models

## Title

Interpreting Unsupervised Learning Models

## Dataset

Iris Dataset (https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).

I chose this dataset because it is clean, compact, but still it presents two classes that are difficult to separate.

## Functions

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
```

## Instructions

1) As the new researcher in a pharmaceutical company, one of the most promising developments involves extracting the native oils from Iris flowers. Lucky for us, we have a previous dataset of Iris flowers (all 3 species) with measurements of their sepal and petal lengths and widths. Your task is to develop a method to distinguish them for any future flower.

    - Run the `load_iris` method  to load both the data and true labels as dataframes. I recommend naming them `data` and `target`.
    - Verify that both the `data` (the first element from the output of `load_iris`) as well as `target` have 150 rows
    - Standardise our data by using the StandardScaler and `fit_transform` on `data`.
    

2) Now we are ready for doing some machine learning! In particular, for now we will forget we have the true values, in order to use them for calibration afterwards. So we will just use the scaled `data` DataFrame (or Numpy array if you did not convert it back). 
   Our strategy will be to use clustering in order to be able to figure out on new flowers which is the most probable species. For that first we need to find out the number of clusters
   
    - Instantiate a KMeans estimator with `init="random"` and `n_clusters` whatever you prefer, here we will get used to this estimator 
    - Fit the estimator with the `data` dataset
    - Print the `inertia_` attribute of the `Kmeans` fitted estimator. This number is a measurement of how far are the points from the cluster center. We want it as small as possible
    - Finally, print and calculate the `silhouette_score` output passing as input to this function the `data` dataset, the estimated labels and `metric=euclidean`. This number is a measurement of how often a value is placed into a wrong cluster, it goes from -1 to 1, and closer to 1 is better
    


3) Now that we are familiar with fitting a KMeans estimator and its metrics, let's find the optimal `n_clusters`.

    - Create a loop for `n_clusters` between 2 and 10 where it will:
        * Instantiate a KMeans estimator with `n_clusters` the one from the loop. I suggest setting `init="random"` and `n_init="auto"`
        * Fit the estimator with the data
        * Store the `inertia_` attribute in a list, for example `inertia_results`
        * Store the `silhouette_score` in another list, for example `silhouette_results`
    - Using the following command as help `f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))`:
        * Plot on the first axis the `inertia_results` as a function of `n_clusters`
        * Plot on the first axis the `silhouette_results` as a function of `n_clusters`
    - There is a rule in clustering called the "Elbow rule", which says that the `n_clusters` such that the inertia produces an "elbow", is the optimal. Do you see one like that in the inertia plot?
    - If, on the other hand, we are checking the Silhouette Score, which `n_clusters` provides the best score?
    
    On both criteria you should get the best `n_clusters` is 3. And indeed in this dataset we have only 3 species of Iris.


4) We have the optimal number of clusters, let's try to visualize this result. To do this we need to go from 4 dimensions into 2 dimensions, this invokes a technique called "Dimensionality Reduction"

    - Instantiate a `PCA` object specifying we want to keep 2 `n_components`. This is the dimension we want in the end.
    - Use the `fit_transform` method of the `PCA` estimator to pass our `data` and get back a `reduced_data` Numpy Array.
    - Fit a `KMeans` estimator with `n_clusters=3` as before, but passing `reduced_data`
    - Run a scatter plot setting `x` to be `reduced_data[:,0]`, `y` would be `reduced_data[:,1]` and `c` would be the prediction of `Kmeans` on the `reduced_data`, this would color each cluster differently.
    
    What does this plot tell you? Is KMeans able to split your data correctly?
    
    - If you want to visually check if this is OK, you can do another plot as before, but changing `c` to be `target`, which are the true values.
    


5) Another technique to add to your repertoire of Dimensionality Reduction tools is `t-SNE`. It is helpful on nonlinear data (not this case, but if you encounter a dataset where the boundary is nonlinear this will help you). To use it is very similar to `PCA` but we need to tune the `perplexity`.

    - Create a loop of perplexity values from 10 to 60 every 5, and on that loop: 
        - Instantiate a `TSNE` object specifying we want to keep 2 `n_components`. This is the dimension we want in the end.
        - Use the `fit` method of the `TSNE` estimator passing our `data`.
        - Save the `kl_divergence_` attribute
    - Plot the `kl_divergence_` vs `perplexity` and choose the perplexity where the `kl_divergence_` is constant (i.e.,: "converged")    
    - Instantiate a `TSNE` object specifying we want to keep 2 `n_components` and `perplexity` the one you chose before.
    - Use the `fit_transform` method of the `TSNE` estimator to pass our `data` and get back a `reduced_data` Numpy Array.
    - Fit a `KMeans` estimator with `n_clusters=3` as before, but passing `reduced_data`
    - Run a scatter plot setting `x` to be `reduced_data[:,0]`, `y` would be `reduced_data[:,1]` and `c` would be the prediction of `Kmeans` on the `reduced_data`, this would color each cluster differently.



In this short project we have been able to use Dimensionality Reduction and Clustering techniques to understand our data, both in linear and nonlinear case, and how to treat it for downstream applications.