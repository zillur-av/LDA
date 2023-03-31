# Introuduction

Linear Discriminant Analysis(LDA) is a machine learning classification algorithm. In this repository, we implement this model from scratch (no built in library). We use [Raisin Dataset](https://archive.ics.uci.edu/ml/datasets/Raisin+Dataset) from UCI Machine Learning Repository. The dataset has 2 classes and 7 features.

# Implementation Details
First, clone the repo using the following command line:

```git clone https://github.com/zillur-av/LDA.git```

# Linear Discriminant Analysis from Scratch in Python
This project is an implementation of Linear Discriminant Analysis (LDA) from scratch in Python. LDA is a classification algorithm used in machine learning that finds a linear combination of features that best separates two or more classes of objects or events. The algorithm is used extensively in pattern recognition and has applications in fields such as image and speech recognition.

# Algorithm
LDA works by finding a linear combination of features that maximizes the ratio of the between-class scatter to the within-class scatter. The goal is to find a decision boundary that best separates the classes, by minimizing the overlap between the different classes while maximizing the distance between them.

The algorithm consists of the following steps:

* Calculate the mean vector for each class.

* Calculate the within-class scatter matrix `S_w`.

* Calculate the between-class scatter matrix `S_b`.

* Compute the optimal weights `w` and bias term `w_0` that maximize the ratio of `S_b` to `S_w`.

* Use the decision boundary defined by `w` and `w_0` to classify new examples.

# Mathematical Equations
The following equations are used in the implementation:

### Mean vector:
$$\mu_c = \frac{1}{n_c}\sum_{i=1}^{n_c} x_i$$
where `n_c` is the number of examples in class c, `x_i` is the i-th example, and `\mu_c` is the mean vector for class c.

### Within-class scatter matrix:
$$S_w = \sum_{i=1}^{n_0}(x_i - \mu_0)(x_i - \mu_0)^T + \sum_{i=1}^{n_1}(x_i - \mu_1)(x_i - \mu_1)^T + ... + \sum_{i=1}^{n_C}(x_i - \mu_C)(x_i - \mu_C)^T$$
where `n_c` is the number of examples in class c, `x_i` is the i-th example, `\mu_c` is the mean vector for class c, C is the total number of classes, and T denotes the transpose of a matrix.

### Between-class scatter matrix:
$$S_b = \sum_{c=1}^C n_c(\mu_c - \mu)(\mu_c - \mu)^T$$
where `n_c` is the number of examples in class c, `\mu_c` is the mean vector for class c, C is the total number of classes, and `\mu` is the mean vector for all classes.

### Linear discriminant function:
$$y(x) = w^Tx + w_0$$
where x is the input example, w is the weight vector, and w_0 is the bias term.

### Optimal weights:
$$w = S_w^{-1}(\mu_1 - \mu_0)$$
where `S_w` is the within-class scatter matrix, and `\mu_0` and `\mu_1` are the mean vectors of the two classes.

### Optimal bias:
$$w_0 = -w^T\frac{(\mu_1 + \mu_0)}{2}$$
where w is the weight vector, and `\mu_0` and `\mu_1` are the mean vectors of the two classes.


