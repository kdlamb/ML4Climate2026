# Unsupervised Machine Learning

Unsupervised ML refers to learning from unlabeled data sets. That is, we try to learn relationships between points in our data based on the features themselves. In this section, we will:
- Understand the necessity of reducing dimensionality for large datasets.
- Learn about different approaches for dimensionality reduction
- Discuss Principal Component Analysis (PCA)
- Learn about different approaches for clustering
- Learn about K-Means Clustering, DBSCAN, and Gaussian Mixture Models
- Discuss Applications of Unsupervised Machine Learning to Environmental Science and Climate Science

## Dimensionality Reduction

Dimensionality reduction refers to techniques that reduce the number of features in a dataset, while retaining essential information. It can be used for a number of downstream tasks, such as data visualization and analysis, feature engineering, and improved model perfomance.

High-dimensional data sets are often very sparse, with training instances very far away from one another. The curse-of-dimensionality is an idea that adding more variables, or dimensions, to a data set can lead to increased sparsity, greater computational complexity, and decreased efficiency in applying machine learning methods to these data sets. However, high dimensional data sets often have a much smaller set of possible states, meaning in practice the degrees of a freedom of a data set are actually quite limited.

Dimensionality reduction methods help us to discover, and exploit, these lower dimensional relationships.

Some of the main methods for dimensionality reduction are:
- **Projection methods**: methods that linearly transform data to lower dimensions (e.g. Principal Component Analysis, PCA)
- **Manifold learning methods**: methods that capture the underlying data structure for non-linear relationships (e.g. t-Distributed Stochastic Neighbor Embedding, Uniform Manifold Approximation and Projection, etc.)
- **Deep Learning methods**: methods that non-linearly transform data to lower dimensions (e.g. auto-encoders, variational auto-encoders)

## Dimensionality Reduction in Climate and Environmental Science

Dimensionality reduction can address several important challenges for climate and environmental science. Because climate data sets are often high dimensional, an important challenge is how to find interpretable information and patterns in these data sets. The climate consists of many complex, interacting processes that happen on a variety of temporal and spatial scales. For example, spatio-temporal patterns such as the ENSO or the Madden-Julian Oscillation can help us to better understand and predict seasonal weather patterns. Dimensionality reduction can also help with feature engineering and reducing computational expense.

## Clustering Algorithms

Clustering algorithms discover underlying patterns and relationships between data, and can be used for unsupervised classification. In contrast to supervised learning, which requires pre-labeled data sets, clustering algorithms attempt to group data into clusters based on similarity, without pre-defined labels. These methods can be used for data analysis, dimensionality reduction, and anomaly detection (i.e. finding unusual or anomalous samples in the data set).

There are many different types of clustering algorithms. Some common ones are K-Means clustering, DBSCAN, and Gaussian Mixture Models.