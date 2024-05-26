# ML-Wine-Quality-Prediction

This is a project of a graduated level machine learning course that aims to classify good/bad wine base on costum setting (by the user) on the wine quality. The reason behind this is simple: common people might not be able to judge how good is a bottle of wine by quantifying all the wine features such as sulphur oxides or different types of acidity (like a professonal wine taster!). However, if a classification model is set up by the user itself, this might help people to roughly know which new wine might fit their need, or at least rule out the new wine they might not like in the future.

The project was originally finished in R (check it out!) but it was improved and written in python language (shown here).

The description and the datasets can be accessed here:

https://archive.ics.uci.edu/dataset/186/wine+quality

Please note that only **white** wine data is used here.

The project will be presented into two parts: 1. Binary classification. 2. Mulitclass Classification

For each part, in general, the structure follows:

**1. Data Access and threshold setting**

**2. Data Visualization**
   
**3. Data Preparation**
   
**4. Logistic Regression Approach**   

**5. Random Forest Approach**   

**6. Support Vector Classifier Approach**


# 1. Binary Classification

As you can see, we will use different models and see which one performs the best. But first we need to access the data based on the user's interest.

One of the feature is "quality" (from 1 to 10, where 10 is perfect) of the wine. Assume that the customer has normal drinking habit and basic knowledge on different wine, he/she may have an idea on what "standard" (i.e quality) he/she is looking for. Simply, the customer can set a threshold on this feature to distinguish the "good/bad" label of the wine.

First of all, let's type in a number and set the threshold:
![alt text](images/1.png)

In this demonstration, we set the quality standard as 6, which means every wine higher than 6 is considered good wine. Here are all the features:
![alt text](images/1a.png)

And their statistical information:
![alt text](images/1b.png)

Now we apply a new feature "judge" to indicate the wine label. This feature is binary (0 = bad wine & 1 = good wine) and it depends on the "quality":
![alt text](images/1c.png)

According to the threshold, distributions can be visualized in every features. First of all, let's see how many "bad" and "good" wine via a pie chart:
![alt text](images/1d.png)
![alt text](images/pie_bad_good_wine.png)

The distributions of continuous features can be visualized via histograms. A for-loop is used to serve the purpose:
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

The density, meaning the texture of the wine:
![alt text](images/con_density.png)

And the concentration of alcohol:
![alt text](images/con_alcohol.png)

The only caterogrical feature is the quality of the wine:
![alt text](images/1f.png)
![alt text](images/cat_quality.png)

## Data Preparation

Each instance in the dataset (total 4898 instance) is randomly assigned a number (called "sample") for data splitting: 4000 for training and 898 for testing. Also, "quality" and "sample" were removed as they are not used for training and testing.
![alt text](images/1g.png)

Now we can start training the models. Let's start with the Logistic Regression Approach.

## Logistic Regression (LR) Approach

The Logistic Regression is a common method for binary classification. As we want to keep all parameters, Ridge Regression, instead of LASSO, is chosen for the regulariztion to prevent overfitting. Also, the maximum iteration is set to be 1500 since it is not too computational expensive.

After that the fitting was performed, a prediction is made with the test set.
![alt text](images/1h.png)

For binary classification, it is convinent to use the confusion matrix to visualize the result:
![alt text](images/1i.png)
![alt text](images/cm_lr1.png)

Sometimes the probability threshold for each verdict may not be 0.5 (For example, say over 70% probability then the wine is considered as "good"). The ROC curve, which gives us an idea of all confusion matrics for any probability threshold:
![alt text](images/1k.png)
![alt text](images/roc_lr1.png)

But in our case, probability threshold is set to be normal (50%), so the confusion matrix above is enough. By looking at the matrix elements, it seems that the Type II error (False Negative) is high.

But please be patient, let's look at other methods.

## Random Forest (RF) Approach

Random Forest approach is the second method for this classification task. The idea of Random Forest method is simply create a set of simple decision trees (or so-called "weak learners"), which each tree is created (with its own fitting) with a subset of explanatory variables. Each tree is then applied to the testing set and make prediction on all instances. For each instance, voting from all trees would make a final decision (classification!) for the instance.

1500 trees are created for training and testing and $\sqrt{n}$ features are used for tree creation. Gini impurity is selected for splitting a node: it is because the computational cost is lower than using entropy; and minimum split is 2 for the nodes due to the same reason.

![alt text](images/1l.png)

As per the result, below is the confusion matrix:
![alt text](images/1m.png)
![alt text](images/cm_rf1.png)

The ROC curve:
![alt text](images/1o.png)
![alt text](images/roc_rf1.png)

The confusion matrix shows us that RF does a better job than the LR approach! How about the last approach?

## Support Vector Classifier (SVC) Approach

The last method is to use Support Vector Classifier with a non-linear boundary. The idea of SVC is to make use of hyperplane for classification. If a dataset contains p variables, all instances can be seen as p-dimensional vectors with (p+1) coefficients. To classify points in a p-dimensional space, a hyperplane, which is a flat affine subspace of (p-1)-dimensions (the feature space), is defined and used with reasonable bias-variance balancing.

In our case, this is a one-vs-one classification. A non-linear hyperplane is assumed and radial basis function (rbf) kernal is selected for more flexible situation. On the other hand, the gamma is set to be default: $1/(n_features * X.var())$ to control the influence of each instance on the decision boundary, or simply, prevent overfitting.

![alt text](images/1p.png)

Surprisingly, the confusion matrix shows a zero TP result:
![alt text](images/1q.png)
![alt text](images/cm_svc1.png)

And the corresponding ROC curve:
![alt text](images/1s.png)
![alt text](images/roc_svc1.png)

## Combined results

In order to choose the best methods (among three), the confusion matrics are shown side-by-side:
![alt text](images/1t.png)
![alt text](images/cm_all1.png)

By just looking at the matrics, it seems that Random Forest model obtains a better result, meaning the diagonal elements has more counts in RF model than the other two models.

Also, their performance is printed out for comparison. The LR and SVC methods show similar performance: the area under the curve (AUC), prediction accuracy and error rate are ~0.75, ~0.78 and ~0.22.

The RF method has the best performance with AUC of ~0.93, prediction accuracy of ~0.87 and error rate of ~0.13.
![alt text](images/1u.png)

And it is clear by observing their ROC curves:
![alt text](images/1v.png)
![alt text](images/roc_all1.png)

It is obvious that, the RF's ROC curve (red) is close to the top left corner, which means a perfect classification model.

To conclude, we can see that the RF model gives out the best performance. Now we should move to next stage: Multi-class Classification.


# 2. Multi-class Classification

Of casue, sometimes people may want to choose wine based on their situation. For example, people may purchase different types of wine as gifts for friends and family during Christmas. The multi-class classification would be helpful for this purpose.

In this part, we implant two arbitrary thresholds, that is, two boundaries are set to classify all the wine into "bad", "normal" and "good" category. Of cause, in reality, things can be different. For example, wine retailers can set up the price of a new wine wines, by classifying it into: "cheap", "normal" or "expensive" based on the dataset or variables.

Same as before, first step is to insert the thresholds. In this example, "quality" below 5 are considered as "bad" wine and above 6 are "good" wine. Any wine with quality between (i.e 5 and 6) are "normal" wine.
![alt text](images/2a.png)

After inserting the thresholds, a logic gate would apply to check if there is(are) any contradiction(s). For example, the lower threshold cannot have a higher quality than the higher threshold.
![alt text](images/2b.png)

Displaying statistical information of the dataset:
![alt text](images/2c.png)

Labels "0", "1" and "2" for each wine based on the thresholds.
![alt text](images/2d.png)

Lets have a quick look on the categories with a simple pie chart. About 74.6% of wine are classified as "Normal", 21.6% of them as "Good" and only 3.7% as "Bad".
![alt text](images/2e.png)
![alt text](images/pie_bad_nor_good_wine.png)

Splitting the dataset into training and testing set, with the size of both training and test set are the same as before:
![alt text](images/2f.png)

The methods used and shown here are Logistic Regression and Random Forest, as we have seen that SVC has similar performance as Logistic Regression and we want to lower the computational cost.
![alt text](images/2h.png)

After training, the models' prediction on the test set can be shown in the confusion matrics:
![alt text](images/2i.png)
![alt text](images/cm_all2.png)

Still, we can see that, the Random Forest has the best performance, with ~0.852 of accuracy and ~0.148 error rate:
![alt text](images/2j.png)

************************************************************************************************************************

# Conclusion, Business Insight and Improvement

To conclude, this binary/ multi-class classification system can help customer to have a judgement on a new wine. The Random Forest model shows a better result than that of Logistic Regression and Support Vector Classifier.

It is recommended that the classification system should be applied and run with an online ordering system: each client can build their own dataset and the sytem can provide a more accurate recommandation via purchasing.

Furthermore, this classification system, is not bounded to this use. As mentioned, wine retailers can also use the system to help them decide the price of a new wine. Another example would be to help wine producers to decide which wine should be produced and export more, based on the dataset built by their clients.

This classifier can be improved and designed with using different types of wine, instead of using single wine type. Since wine types are produced differently, the number of features we used here should be expanded. Also, as the chemical contents might be different for different types (for example, sulphur dioxide of red wine might be higher than that of beer), proper scaling and statistical method might be needed to pre-process the dataset before model training.
