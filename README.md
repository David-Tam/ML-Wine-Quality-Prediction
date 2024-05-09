# ML-Wine-Quality-Prediction
This is a project of a graduated level machine learning course that aims to classify good/bad wine base on costum setting (by the user) on the wine quality. The project was originally finished in R (check it out!) but it was improved and written in python language (shown here).

The description and the datasets can be accessed here:

https://archive.ics.uci.edu/dataset/186/wine+quality

Please not that only **white** wine data is used here.

The project will be presented into two parts: 1. Binary classification. 2. Mulitclass Classification
In general, the structure follows:

**1. Data Access and threshold setting**

**2. Data Visualization**
   
**3. Data Preparation**
   
**4. Logistic Regression Approach**   

**5. Random Forest Approach**   

**6. Support Vector Classifier Approach**


# 1. Binary Classification

The feature of this project is returning "good/bad" label for each wine, and the "standard" is set by the user. The reason behind is simply: common people might not be able to quantify the quality for a bottle of wine (like a professonal wine taster!), this might help them to roughly know which wine might fit their need in the future.

One of the feature is "quality" (from 1 to 10, where 10 is perfect) of the wine, we simply set a threshold on this feature to distinguish the label of the wine.

First of all, we type in a number and set the threshold:
![alt text](images/1.png)

Let's say every quality higher than 6 is considered good wine. Here are all the features:
![alt text](images/1a.png)

And their statistical information:
![alt text](images/1b.png)

Now we apply a new feature "judge" to indicate the wine label. This feature is binary (0 means bad wine and 1 is good wine) and depends on the "quality":
![alt text](images/1c.png)

According to the threshold, distributions can be visualized in every features. First of all, let's see how many "bad" and "good" wine via a pie chart:
![alt text](images/1d.png)
![alt text](images/pie_bad_good_wine.png)

The distributions of continuous features can be visualized via histogram. A for-loop is used to serve the purpose:
![alt text](images/1e.png)

For acidity-related features:
![alt text](images/con_fixed_acidity.png)
![alt text](images/con_volatile_acidity.png)
![alt text](images/con_citric_acid.png)
![alt text](images/con_pH.png)

How sweet is the wine?
![alt text](images/con_residual_sugar.png)

How about chlorides?
![alt text](images/con_chlorides.png)

The sulphur dioxide features:
![alt text](images/con_free_sulfur_dioxide.png)
![alt text](images/con_total_sulfur_dioxide.png)
![alt text](images/con_sulphates.png)

The density, meaning the texture of the wine?
![alt text](images/con_density.png)

And the concentration of alcohol:
![alt text](images/con_alcohol.png)

The only caterogrical feature is the quality of the wine:
![alt text](images/1f.png)
![alt text](images/cat_quality.png)

### Data Preparation

The whole data set (4898 data point) is assigned a random number (called "sample") for data splitting: 4000 for training and 898 for testing. Then, "quality" and "sample" were removed from both sets as they are not used for training and testing.
![alt text](images/1g.png)

## Logistic Regression (LR) Approach

First method attemped is the Logistic Regression, which is quite a common method for binary classification. Since it is not clear how close are the values for each parameter, Ridge Regression is chosen for the regulariztion to prevent overfitting. As this is not too computational expensive, maximum iteration is set to be 1500.

After that the fitting was performed and made a prediction with the test set.
![alt text](images/1h.png)

First, let's check the confusion matrix:
![alt text](images/1i.png)
![alt text](images/cm_lr1.png)

And the corresponding ROC curve, which gives us an idea of all confusion matrics for any probability threshold:
![alt text](images/1k.png)
![alt text](images/roc_lr1.png)

## Random Forest (RF) Approach

Random Forest approach is the second method for this classification task. The idea of Random Forest method is simply create a set of simple decision trees (or so-called "weak learner"), which each tree is created (with its own fitting) with a subset of parameters within the training set. Each tree is then applied to the testing set and make prediction on all data. For each data, voting from all trees would make a final decision, or classification, for a data point.

1500 trees are created for training and testing and $\sqrt{n}$ features are used for tree creation. Gini impurity is selected as the computational cost is lower than using entropy; and minimum split is 2 for the nodes due to the same reason.

![alt text](images/1l.png)

As per the result, below is the confusion matrix:
![alt text](images/1m.png)
![alt text](images/cm_rf1.png)

The ROC curve:
![alt text](images/1o.png)
![alt text](images/roc_rf1.png)

## Support Vector Classifier (SVC) Approach

The last method is to use Support Vector Classifier with a non-linear boundary. The basic idea is to make use of hyperplane to classify. If a dataset is of p variables, all data points can be seen as p-dimensional vectors with (p+1) coefficients. To classify points in a p-dimensional space, a hyperplane, which is a flat affine subspace of (p-1)-dimensions (a feature space), is defined and used with reasonable bias-variance balancing.

For this one-vs-one classification, non-linear hyperplane is assumed, and radial kernal is selected for more flexible situation. The gamma is set to be default: $1/(n_features * X.var())$ to prevent overfitting.

![alt text](images/1p.png)

The confusion matrix shows a zero TP result:
![alt text](images/1q.png)
![alt text](images/cm_svc1.png)

Below is the ROC curve:
![alt text](images/1s.png)
![alt text](images/roc_svc1.png)

## Combined results

In order to compare results from the methods, the confusion matrix is shown side-by-side:
![alt text](images/1t.png)
![alt text](images/cm_all1.png)

Also, the performance are printed out for comparison. The LR and SVC methods show similar performance: the area under the curve (AUC), prediction accuracy and error rate are ~0.75, ~0.78 and ~0.22.

The RF method has the best performance with AUC of ~0.93, prediction accuracy of ~0.87 and error rate of ~0.13.
![alt text](images/1u.png)

And it is clear to obseve the comparison with the ROC curves:
![alt text](images/1v.png)
![alt text](images/roc_all1.png)




# 2. Multi-class Classification

The other attempt is to perform multi-class classification with two arbitrary thresholds. That is, we set two boundaries and classify all the wine into "bad", "normal" and "good" category.

Firstly, let's insert the thresholds using the quality. In this example, below 5 are considered as "bad" wine and above 6 are "good" wine. Any wine with quality between (i.e 5 and 6) are "normal" wine.
![alt text](images/2a.png)

After inserting the thresholds, a logic gate would apply to check if there is(are) any contradiction(s). For example, the lower threshold cannot have a higher quality than the higher threshold.
![alt text](images/2b.png)

Displaying statistical information of the dataset:
![alt text](images/2c.png)

Assigning label for each category, according to the thresholds inserted:
![alt text](images/2d.png)

Lets have a quick look on the categories with a simple pie chart:
![alt text](images/2e.png)
![alt text](images/pie_bad_nor_good_wine.png)

Data splitting into training and testing set:
![alt text](images/2f.png)

The method we used are Logistic Regression and Random Forest, as we see that SVC has similar performance as Logistic Regression and we want to lower the computational cost.
![alt text](images/2h.png)

After training, the prediction on the test set can be shown in the confusion matrics:
![alt text](images/2i.png)
![alt text](images/cm_all2.png)


![alt text](images/2j.png)
